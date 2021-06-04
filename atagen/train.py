import argparse
import functools
import os
from pfrl.initializers.chainer_default import init_chainer_default
from pfrl.q_functions.state_q_functions import DiscreteActionValueHead

import torch
import numpy as np
import pfrl
from pfrl import nn as pnn
from pfrl import experiments
from pfrl.wrappers import atari_wrappers


class TrainTrial():
	def __init__(self, test=False) -> None:
		self.args = self.parse_args()
		if test:
			self.args.steps = 50000
		self.setup()

	def parse_args(self):
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument("--env_name", type=str, default="MsPacmanNoFrameskip-v4")
		parser.add_argument("--results-dir", type=str, default="results",
				help="Directory path to save output files. If it does not exist, it will be created.")
		parser.add_argument("--tag", type=str, default="default_exp", 
				help="tag for the specific experiment run, also used as the subdir for saving")
		parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
		parser.add_argument("--gpu", type=int, default=0)
		parser.add_argument("--load", type=str, default=None)
		parser.add_argument("--final-exploration-frames", type=int, default=10 ** 6)
		parser.add_argument("--final-epsilon", type=float, default=0.01)
		parser.add_argument("--eval-epsilon", type=float, default=0.001)
		parser.add_argument("--steps", type=int, default=5 * 10 ** 7)
		parser.add_argument(
			"--max-frames",
			type=int,
			default=30 * 60 * 60,  # 30 minutes with 60 fps
			help="Maximum number of frames for each episode.",
		)
		parser.add_argument("--replay-start-size", type=int, default=5 * 10 ** 4)
		parser.add_argument("--target-update-interval", type=int, default=3 * 10 ** 4)
		parser.add_argument("--eval-interval", type=int, default=10 ** 5)
		parser.add_argument("--update-interval", type=int, default=4)
		parser.add_argument("--eval-n-runs", type=int, default=10)
		parser.add_argument("--no-clip-delta", dest="clip_delta", action="store_false")
		parser.set_defaults(clip_delta=True)
		parser.add_argument(
			"--agent", type=str, default="DoubleDQN", choices=["DQN", "DoubleDQN", "PAL"]
		)
		parser.add_argument(
			"--log-level",
			type=int,
			default=20,
			help="Logging level. 10:DEBUG, 20:INFO etc.",
		)
		parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate")
		parser.add_argument(
			"--prioritized",
			action="store_true",
			default=False,
			help="Use prioritized experience replay.",
		)
		parser.add_argument("--num-envs", type=int, default=4)
		parser.add_argument("--n-step-return", type=int, default=1)

		args = parser.parse_args()
		return args
	
	def setup(self):
		# logging
		import logging
		logging.basicConfig(level=self.args.log_level)

		# set a random seed used in PFRL
		pfrl.utils.set_random_seed(self.args.seed)
		# Set different random seeds for different subprocesses.
		# If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
		# If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
		process_seeds = np.arange(self.args.num_envs) + self.args.seed * self.args.num_envs
		assert process_seeds.max() < 2 ** 32
		self.process_seeds = process_seeds

		# gpu or cpu
		if self.args.gpu == 0 and not torch.cuda.is_available():
			self.args.gpu = -1
			logging.info("training on device CPU")
		else:
			torch.backends.cudnn.benchmark = True
			logging.info("training on device CUDA")

		# output
		# self.args.outdir = experiments.prepare_output_dir(self.args, os.path.join(self.args.results_dir, self.args.tag))
		self.args.outdir = os.path.join(self.args.results_dir, self.args.tag)
		logging.info('Output files are saved in {}'.format(self.args.outdir))

	def build_agent(self):
		# q function
		sample_env = self.make_env(0, test=False)
		n_actions = sample_env.action_space.n
		self.q_func = torch.nn.Sequential(
			pnn.LargeAtariCNN(),
			init_chainer_default(torch.nn.Linear(512, n_actions)),
			DiscreteActionValueHead(),
		)
		# optimizer
		optimizer = torch.optim.RMSprop(
			self.q_func.parameters(),
			lr=self.args.learning_rate,
			alpha=0.95,
			momentum=0.0,
			eps=1e-2,
			centered=False,
		)

		# Select a replay buffer to use
		if self.args.prioritized:
			# Anneal beta from beta0 to 1 throughout training
			betasteps = self.args.steps / self.args.update_interval
			replay_buffer = pfrl.replay_buffers.PrioritizedReplayBuffer(
				10 ** 6,
				alpha=0.6,
				beta0=0.4,
				betasteps=betasteps,
				num_steps=self.args.n_step_return,
			)
		else:
			replay_buffer = pfrl.replay_buffers.ReplayBuffer(10 ** 6, num_steps=self.args.n_step_return)

		# feature extraction
		def feature_extractor(x):
			return np.asarray(x, dtype=np.float32) / 255

		# explorer
		explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
			1.0,
			self.args.final_epsilon,
			self.args.final_exploration_frames,
			lambda: np.random.randint(n_actions),
		)

		# construct agent
		def parse_agent(agent):
			return {"DQN": pfrl.agents.DQN, "DoubleDQN": pfrl.agents.DoubleDQN, "PAL": pfrl.agents.PAL}[agent]
		Agent = parse_agent(self.args.agent)
		agent = Agent(
			self.q_func,
			optimizer,
			replay_buffer,
			gpu=self.args.gpu,
			gamma=0.99,
			explorer=explorer,
			replay_start_size=self.args.replay_start_size,
			target_update_interval=self.args.target_update_interval,
			clip_delta=self.args.clip_delta,
			update_interval=self.args.update_interval,
			batch_accumulator="mean",
			phi=feature_extractor,
		)
		return agent

	
	def make_env(self, idx, test):
		# Use different random seeds for train and test envs
		process_seed = int(self.process_seeds[idx])
		env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
		env = atari_wrappers.wrap_deepmind(
			atari_wrappers.make_atari(self.args.env_name, max_frames=self.args.max_frames),
			episode_life=not test,
			clip_rewards=not test,
			frame_stack=False,
		)
		if test:
			# Randomize actions like epsilon-greedy in evaluation as well
			env = pfrl.wrappers.RandomizeAction(env, self.args.eval_epsilon)
		env.seed(env_seed)
		return env
	
	def make_batch_env(self, test):
		vec_env = pfrl.envs.MultiprocessVectorEnv(
			[
				functools.partial(self.make_env, idx, test)
				for idx, env in enumerate(range(self.args.num_envs))
			]
		)
		vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
		return vec_env
	
	def run(self):
		agent = self.build_agent()

		if self.args.load:
			agent.load(self.args.load)
		
		pfrl.experiments.train_agent_batch_with_evaluation(
			agent=agent,
			env=self.make_batch_env(test=False),
			eval_env=self.make_batch_env(test=True),
			steps=self.args.steps,
			eval_n_steps=None,
			eval_n_episodes=self.args.eval_n_runs,
			eval_interval=self.args.eval_interval,
			outdir=self.args.outdir,
			save_best_so_far_agent=False,
			log_interval=10000,
		)


def main(test=False):
	trial = TrainTrial(test=test)
	trial.run()


if __name__ == "__main__":
	main()

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .core import QueryStrategy, Model
import numpy as np


# In[ ]:


## An adaptive active learning algorihm. Designed to solve a Multi armed bandit with expert advice problem (MAB) with a non stationary reward of different experts.
## For more info see References of main class.

class DynamicEnsembleActiveLearning(QueryStrategy):
    '''
    A QueryStrategy object designed to adaptively manage the advice of different underlying QueryStrategy objects.
    In the context of Multi armed bandit problems, the bandit has the choice of K arms being the unlabeled sample and receiving a reward after the queried sample is labeled.
    The N experts (active learners) each give their vote for every action and the total vote for a certain action/arm is the sum of votes weighted by the experts respective weight/saying.
    After receiving the reward the weight of every expert gets updated proporional to their respective vote in that action - several experts may have adviced the same arm.
    Then the next round starts ..


    Parameters
    ----------

    query_strategy: list, shape = (n_active_learners)
        Core active learners, iterable list of QueryStrategy objects

    model: Model
        Main Classifier that is trained.

    dataset: Dataset

    gamma: float (0,1], hyperparameter controlling learning rate (default = 0.1)

    delta_T: int, hyperparameter: number of iterations before experts weight are reset to 1. (default = 10)

    alpha: float, hyperparameter for Gibbs measure (default=0.1)

    beta: float, hyperparameter for Gibbs measure (default=10)


    Attributes
    ----------
    .past_reward: {float}, Last IWAcc

    .rexp4.weights: {np.array}, shape = (n_experts)
        Weights of the individual experts


    Methods
    -------

    .make_query():
        returns
        entry_id: int, id of the sample to label


    References
    ----------

    [1] Dynamic Ensemble Active Learning: A Non-Stationary Bandit with Expert Advice. Sept 2018
        Kunkun Pang, Mingzhi Dong, Yang Wu, Timothy M. Hospedales

    [2] Active Learning by Learning. 2015
        Wei-Ning Hsu, Hsuan-Tien Lin
    '''

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        self.qs = kwargs.pop('query_strategy', None)
        ## Sanity checks
        if self.qs is None:
            raise TypeError('You have to give me QueryStrategies! query_strategy should be something')

        for qs in self.qs:
            if not isinstance(qs, QueryStrategy):
                raise TypeError('One of your active learners is not a QueryStrategy object')


        self.model = kwargs.pop('model', None)
        ## Sanity checks
        if not isinstance(self.model, Model):
            raise TypeError('Your model must be of type Model')

        ## Fit it once
        self.model.train(self.dataset)

        ## Learning rate
        self.gamma = kwargs.pop('gamma', 0.1)

        ## Maximal number of iterations
        self.T = kwargs.pop('T', None)
        if self.T is None:
            raise ValueError('T must be given.')

        ## Number of iterations before weights reset
        self.delta_T = kwargs.pop('delta_T', 10)

        ## Gibbs parameters
        self.alpha = kwargs.pop('alpha', 0.1)
        self.beta = kwargs.pop('beta', 10)

        ## Queried history
        self.queried_hist = None
        self.queried_hist_w = None

        ## Initialize Core solver
        self.rexp4 = REXP4(
            experts = self.qs,
            delta_T = self.delta_T,
            gamma = self.gamma,
            alpha = self.alpha,
            beta = self.beta
        )

        ## Cache for last reward
        self.past_reward = None

    def reward(self):
        '''
        IW Accuracy as reward function.
        See [2] for more information.
        '''
        ## Train the model first
        self.model.train(self.dataset)

        ## Calculate the IWAcc
        IW = 0
        ## Number of arms of last question
        K = self.dataset.len_unlabeled()+1
        ## Number of queries before
        l = len(self.queried_hist)

        for i in range(l):
            ## Number of arms at that past history point
            K_hist = K+l-i

            idx = self.queried_hist[i]
            X, y = self.dataset.__getitem__(idx)
            X = np.array([X])
            if self.model.predict(X)[0] == y:
                IW += self.queried_hist_w[i]/K_hist
        return IW/self.T

    def make_query(self):
        if self.queried_hist is None:
            reward = -1
            last_idx = -1

            ## Initialize for appending
            self.queried_hist = []
            self.queried_hist_w = []
        else:
            reward = self.reward()
            last_idx = self.queried_hist[-1]

        ## Receive probabilities for drawing
        query_vector = self.rexp4._next(reward, last_idx)

        ## Draw next idx
        next_idx = np.random.choice(self.rexp4.n, p=query_vector)

        ## Save reward
        self.past_reward = reward

        ## Append to history
        self.queried_hist.append(next_idx)
        self.queried_hist_w.append(1/query_vector[next_idx])

        return next_idx

    def confidence(self):
        pass


class REXP4(object):
    '''
    Core solver. Feed it reward to update weights.
    '''
    def __init__(self, **kwargs):
        self.experts = kwargs.pop('experts')
        self.delta_T = kwargs.pop('delta_T')
        self.gamma = kwargs.pop('gamma')
        self.alpha = kwargs.pop('alpha')
        self.beta = kwargs.pop('beta')
        self._gen = self._gen()

        ## Number of experts
        self.N = len(self.experts)

        ## Cache for weights
        self.weights = None

        ## Number of samples in dataset
        self.n = self.experts[0].dataset.__len__()

    def _next(self, reward, idx):
        if reward == -1:
            return next(self._gen)
        else:
            return self._gen.send((reward, idx))

    def _gen(self):
        ## Initialize weights
        w = np.ones(self.N)

        ## iteration counter for reset mechanism
        count = 1

        while True:
            ## Reset if count exceeds delta_T
            if count > self.delta_T:
                w = np.ones(self.N)
                count = 1

            ## Write weight into cache
            self.weights = w

            ## Number of arms
            K = self.experts[0].dataset.len_unlabeled()

            ## Now every expert casts his votes / gives his ranking (order which to query first)
            score_vector = np.zeros(self.n)

            for qs in self.experts:
                temp_score_vector = np.zeros(self.n)

                ## The entries in ranked order, so first entry is rank 1 and so on
                ranking = qs.make_query(size=K)
                rank = 1
                for entry in ranking:
                    ## Exponential ranking Normalisation
                    temp_score_vector[entry] = -np.exp(-self.alpha * rank)
                    rank +=1
                score_vector = np.vstack((score_vector, temp_score_vector))

            ## Lose the first expert (all zeros)
            score_vector = score_vector[1:]

            ## Gibbs measure
            for n1 in range(self.N):
                for n2 in range(self.n):
                    if score_vector[n1, n2] not in [0]:
                        score_vector[n1, n2] = np.exp(-self.beta * score_vector[n1, n2])

            ## Normalisation w.r.t. actions
            advice_vector = np.array([score_vector[x,:]/np.sum(score_vector[x,:]) for x in range(self.N)])

            ## Calculate total probabilities p for every action
            W = np.sum(w)
            p = (1- self.gamma) * np.dot(w, advice_vector)/W
            for i in range(self.n):
                if p[i] not in [0]:
                    p[i] += self.gamma/K

            ## Receive reward
            #print('p before selection:',p)
            reward, idx = yield p

            ## Rescale reward
            reward = reward/p[idx]

            ## Update weights
            yhat = advice_vector[:, idx] * reward
            #print('reward, advice_vector, yhat, w_before',reward, advice_vector[:,idx],yhat, w)
            w = w* np.exp(self.gamma * yhat /K)
            #print('w_after',w)

            ## Increase counter
            count += 1

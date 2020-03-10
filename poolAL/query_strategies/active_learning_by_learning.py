#!/usr/bin/env python
# coding: utf-8

# In[2]:


from .core import QueryStrategy, Model
import numpy as np
import copy


# In[182]:


class ActiveLearningByLearning(QueryStrategy):
    '''
    An adaptive Active learning algorithm based on Exp4.P, see References.
    Adjusts the weights of several QueryStrategy objects during the querying process.
    A sample is queried based on a weighted majority vote of all QueryStrategy's.

    Parameters
    ----------

    dataset: {Dataset}
        The training dataset.

    query_strategy: {list}
        list of QueryStrategy objects. List of active learners.

    uniform_sampler: {bool}
        If true, includes a Random QueryStrategy.
        default = True

    model: {Model}
        The model used to calculate the IWAcc

    T: {int}
        Number of iterations or label budget
        default = 100

    delta: {float}, in (0,1)
        Hyperparameter
        default = 0.1

    p_min: {float}, in (0,1)
        Hyperparameter, minimal probability of each active learner
        default = np.sqrt(np.log(N))/(K*T) where N = nr_of_active_learners and K=N


    Methods
    -------
    .make_query():
        Returns {int}
        Entry to query next

    References
    ----------
    [1] Contextual Bandit Algorithms with Supervised Learning Guarantees. Alina Beygelzimer and John Langford.
        (Exp4.P is in here)

    [2] Active Learning by Learning. Wei-Ning Hsu and Hsuan-Tien Lin.
        (IWAcc is in here, Reward function)
    '''

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)

        self.qs = kwargs.pop('query_strategy', None)

        ## Number of active learners
        self.N = len(self.qs)

        ## Wether or not to include a random query strategy
        self.uniform_sampler = kwargs.pop('uniform_sampler', True)
        if self.uniform_sampler == True: self.N +=1

        ## Sanity checks for query_strategy
        if self.qs is None:
            raise TypeError('Query_strategy must be given')
        for qs in self.qs:
            if not isinstance(qs, QueryStrategy):
                raise TypeError(
                    'Every query_strategy must be a QueryStrategy object'
                )

        self.model = kwargs.pop('model', None)
        ## Sanity checks for model
        if self.model is None:
            raise TypeError('model must be given')
        if not isinstance(self.model, Model):
            raise TypeError('model must be a Model object')

        ## Number of actions
        self.n = self.dataset.__len__()

        ## In this implementation K=N
        self.K = self.N

        ## Query Budget
        self.T = kwargs.pop('T', 100)

        ## Queried history for Reward function
        self.queried_hist = []
        self.queried_hist_w = []

        ## Parameter
        self.delta = kwargs.pop('delta', 0.1)
        self.p_min = kwargs.pop('p_min', np.sqrt(np.log(self.N))/(self.K * self.T))

        ## Initialize Generator
        self.Exp4P_gen = self.Exp4P()

        ## Reward
        self.past_reward = None

        ## Fit the model once
        self.model.train(self.dataset)

    def Exp4P(self):
        '''
        Core Generator function
        '''

        w = np.ones(self.N)
        while True:

            ## Calculate advice of every expert (query strategy)
            advice = np.zeros(self.n)

            for qs in self.qs:
                temp = np.zeros(self.n)
                temp[qs.make_query()[0]] = 1
                advice = np.vstack((advice, temp))
            if self.uniform_sampler == True:
                mask = ~self.dataset.get_labeled_mask()
                temp = np.array(mask).astype(int)/self.dataset.len_unlabeled()
                advice = np.vstack((advice, temp))
            advice = advice[1:]

            ## Calculate the confidence in experts
            W = np.sum(w)
            p = (1-self.K * self.p_min)* w/W + self.p_min

            ## Calculate combined confidence
            query_vector = np.dot(p, advice)

            ## Draw your action
            entry_id = np.random.choice(self.n, p=query_vector)

            ## Return the winner
            yield entry_id, query_vector[entry_id]

            ## Receive your reward
            reward = self.past_reward

            ## Rescale reward
            rescaled_reward = reward / query_vector[entry_id]

            ## Adjust weights
            yhat = advice[:,entry_id]*rescaled_reward
            vhat = 1/p

            w = w* np.exp(
                self.p_min/2 *(
                    yhat + vhat*np.sqrt(
                        np.log(self.N/self.delta)/(self.K * self.T)
                    )
                )
            )


    def make_query(self):

        ## Calculate the reward of last query
        self.past_reward = self.reward()

        ## Get the action / sample to query
        entry_id, p = next(self.Exp4P_gen)

        ## Append query to history
        self.queried_hist.append(entry_id)
        self.queried_hist_w.append(1/p)

        return entry_id



    def reward(self):
        acc = 0
        if len(self.queried_hist) != 0:
            self.model.train(self.dataset)
            for i in range(len(self.queried_hist)):
                entry = self.queried_hist[i]
                X, y = self.dataset.__getitem__(entry)

                ## If you predict it correctly add weight to IWAcc
                if self.model.predict(np.array([X]))[0] == y:
                    acc += self.queried_hist_w[i]

            return acc/self.T/self.n

        ## The first iteration there is no reward yet.
        else:
            return None

    def confidence(self):
        pass


# In[ ]:

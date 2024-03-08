# POLICY ITERATION ALGORITHM

## AIM:
To implement a policy iteration algorithm for the given MDP.
## PROBLEM STATEMENT
The problem statement is a Five stage slippery walk where there are five stages excluding goal and hole.The problem is stochastic thus doesnt allow transition probability of 1 for each action it takes.It changes according to the state and policy.
## State Space:
The states include two terminal states: 0-Hole[H] and 6-Goal[G]. It has five non terminal states including starting state.

## Action Space:
* Left:0

* Right:1
## Transition Probability:
The transition probabilities for the problem statement is:

* 50% - The agent moves in intended direction.

* 33.33% - The agent stays in the same state.

* 6.66% - The agent moves in orthogonal direction.
## Reward:
To reach state 7 (Goal) : +1 otherwise : 0

## Graphical Representation:

![image](https://github.com/Saibandhavi75/policy-iteration-algorithm/assets/94208895/b3008842-9abc-4a01-a22a-24a1600c517e)

## POLICY ITERATION ALGORITHM:
The algorithm implemented in the policy_iteration is a method used to find the optimal policy in a Markov decision process (MDP). Here's a step-by-step explanation of the algorithm:

1.Initialize the policy pi. In this implementation, a random action is chosen for each state s in the MDP P. The initial policy is represented by the lambda function pi=lambda s:{s:a for s,a in enumerate(random_actions)}[s], where random_actions is a list of randomly chosen actions for each state.

2.Enter a loop that continues until the policy pi is no longer changing. This is determined by comparing the previous policy (old_pi) with the current policy computed in the loop.

3.Store the previous policy as old_pi for comparison later.

4.Perform policy evaluation using the function policy_evaluation. This step calculates the state-values (V) for each state s given the current policy pi. The state-values represent the expected cumulative rewards starting from state s following policy pi and discounting future rewards by a factor of gamma. The function policy_evaluation is called with the arguments pi, P, gamma, and theta.

5.Perform policy improvement using the function policy_improvement. This step updates the policy pi based on the current state-values V. The function policy_improvement is called with the arguments V, P, and gamma.

6.Check if the policy has converged by comparing the previous policy old_pi with the current policy {s:pi(s) for s in range(len(P))}. If they are the same for all states s, the loop is exited.

7.Return the final state-values V and the optimal policy pi.

To summarize, policy iteration iteratively improves the policy by alternating between policy evaluation and policy improvement steps until convergence is reached. The algorithm guarantees to find the optimal policy for the given MDP P with a discount factor gamma.


## POLICY IMPROVEMENT FUNCTION
```
Developed By : D.vishnu vardhan reddy
Reference No : 212221230023

```
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to implement policy improvement algorithm
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state,reward, done in P[s][a]:
          Q[s][a]+= prob*(reward+gamma*V[next_state]*(not done))
          new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi
```

## POLICY ITERATION FUNCTION
```
def policy_iteration(P, gamma=1.0,theta=1e-10):
  random_actions=np.random.choice(tuple(P[0].keys()),len(P))
  pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
  return V,pi
```

## OUTPUT:
## Optimal Policy:
![image](https://github.com/vishnudorigundla/policy-iteration-algorithm/assets/94175324/a2b2fbc6-d6c4-4c7c-a0fd-780c9a9bc078)


## Optimal Value Function:

![image](https://github.com/vishnudorigundla/policy-iteration-algorithm/assets/94175324/f25f4899-500a-449d-b5a2-2c4c46514377)


## Success Rate for Optimal Policy:

![image](https://github.com/vishnudorigundla/policy-iteration-algorithm/assets/94175324/accea8db-afba-4085-9404-6323adcad6e2)



## RESULT:

Thus, a program is developed to perform policy iteration for the given MDP.

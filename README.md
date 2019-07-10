# Cross-Entropy-Method-for-Blackjack
Adaptation of CEM for OpenAI Gym's Blackjack Environment. The code will produce a graph while it is learning. As I worked on this in May 2018, I cannot guarantee that this is still compatible with OpenAI's Gym today.

## Why CEM?
The cross entropy method may seem like a very strange choice for blackjack because CEM can only work in a linearly separable problem space. Of course, in blackjack, players can choose if they want their ace to count for 1 or 11 points. The point of this project was to demonstrate the limitations of a linear solution like CEM in a non-linear setting. The performance of CEM in this setting perfectly illustrates this as it averages out at ~54% losses. Unsurprisingly, despite its limitations, CEM outperforms randomly playing which results in a ~70% loss rate.


## Disclaimer
This code is based on a solution for OpenAI's mountain car problem. I think Karpathy wrote this solution but unfortunately the blog post describing it is no longer available. I converted the code to be compatible with the blackjack setting.

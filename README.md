Throughout this experiment, I found myself experimenting a lot with the hidden layers.
At first, I noticed that more hidden layers required more time to run the experiment
which makes sense. As a result, I mostly stuck with one or two hidden layers. Next, I
tried different amount of convolution & pooling layers. Two or three of those usually
gave me good results in both accuracy and speed. Additionally, I experimented with how
many filters I should create. I tested layers that had the same amount of filters, to
less as you went through the model, to more as you went through the model.

During the process, the biggest issue for me was the amount of neurons in the hidden
layer. I didn't notice significant speed changes when I dramatically increased them,
so I naturally created more of them. However, there were diminishing returns. I thought
most of my success would be in adjusting the single hidden layer I had which didn't
end up doing that much. Additionally, I messed with the dropout rate, but that too
didn't help in a net positive way—if I decreased it, then overall accuracy increased,
but so did the time it took to run, and vice versa.

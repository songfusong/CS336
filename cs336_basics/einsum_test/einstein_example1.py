import torch 
from einops import rearrange, einsum  



## Basic implementation 
Y = D @ A.T 
 # Hard to tell the input and output shapes and what they mean. 
# What shapes can D and A have, and do any of these have unexpected behavior?  
# ## Einsum is self-documenting and robust 
# # D A -> Y 
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")  
## Or, a batched version where D can have any leading dimensions but A is constrained. 
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
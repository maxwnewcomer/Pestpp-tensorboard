#TensorBoard Tools for PEST++ and NWT files#
The goal of this project was to provide users with a more visual understanding of the internal processes within NWT files and PEST model_runs_completed  
---
#####run_tensorboard.py#####
run_tensorboard.py provides users with the ability to see live loss, metric, and weight tracking  
*--inputdir is required when running*  
*--logname is for logname customization*  

**How can you use this resource?**  
- By tracking weights and loss live, you can easily stop a failed run before waste possible hours of computation time
- Easy comparison between runs, allowing for you to tune your models with a visual understanding of your model

#####understand_NWT.py#####
understand_NWT.py allows users to understand how seemingly 'black box' variables effect their model runtime  
*--filepath of NWT_Explore_out.csv needed for visualization*  

**How can you use this resource?**   
- Use the variable selection tools under the *hparam* tab to understand how different combinations of variables change the resulting runtime
- Visualize your variables in both line connection plots and scatterplots comparing variable and runtime, allowing you to extract patterns within your variables

**Recommended**  
Use the useful explore_nwt.py program to compare at least 30 different variable combinations for best results  

####Goals####
[ ] Use machine learning to optimize variable selection, minimizing randomness
[ ] Increased depth support for PEST++, specifically pestpp-ies

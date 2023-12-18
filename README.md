# IntroDuction
<p>
  Since T-SNE visualization is just suitable to signal label dataset , so when you are using multi-label dataset you need to first choose the signal label sample in your Embedding
</p>
<p>
  You can process the embedding using the .mat file format. Here is a simple example to help you deal with T-SNE.py and ensure that your data is in the correct format for input.
</p>



```python
import scipy.io as scio
data = scio.loadmat('0.8839152499129482-64-94-NUS-TDH.mat')
codes = data['r_img']
labels = data['r_l']
import numpy as np
# find signal label featrue
for i in range(21): # NUSWIDE-TC21 has 21 class
    o = np.zeros((21,))
    o[i] = 1
    tot = 0
    wanted_code = []
    for k , j in enumerate(labels):    
        if np.array_equal(o , j): # judge that two vector is completely equal
            tot = tot + 1
            wanted_code.append(codes[k])
            
    print(tot)
    wanted_code = np.array(wanted_code)
    save_label_dist={
        'hash':wanted_code
    }    
    print(wanted_code.shape)
    # print(wanted_label[:10])
    scio.savemat(f'{i}_{wanted_code.shape[0]}.mat',mdict=save_label_dist)
```

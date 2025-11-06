# RNA-Seq Differential Expression (Simple & Easy)

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# Create a small sample dataset (Gene expression values)
df = pd.DataFrame({
    'Gene': ['G1','G2','G3','G4','G5'],
    'Control': [100,200,150,400,120],
    'Treated': [180,150,300,350,250]
})

# Calculate Log2 Fold Change = log2(Treated / Control)
df['Log2FC'] = np.log2(df.Treated / df.Control)

# Label genes as Up, Down, or No Change
df['Status'] = ['Up' if x>1 else 'Down' if x<-1 else 'No Change' for x in df.Log2FC]

# Show results
print(df[['Gene','Log2FC','Status']])

# Plot the results (bar graph)
plt.bar(df.Gene, df.Log2FC, color='skyblue')
plt.axhline(1, c='g', ls='--')     # threshold for upregulation
plt.axhline(-1, c='r', ls='--')    # threshold for downregulation
plt.title("RNA-Seq Differential Expression")
plt.ylabel("Log2 Fold Change")
plt.show()

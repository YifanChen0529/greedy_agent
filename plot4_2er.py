import numpy as np
import matplotlib
import pandas as pd

import matplotlib.pyplot as plt




df_all_pure = []
df_all_greedy = []
for i in range(5):
    df_all_pure.append(pd.read_csv('lio/results/er%d/er4_2_pure_det/log.csv'%(i+1)))
    df_all_greedy.append(pd.read_csv('lio/results/er%d/er4_2_greedy_det/log.csv'%(i+1)))
# n = 5
# df_all_pure.append(pd.read_csv('lio/results/er%d/er4_2_pure_det/log.csv'%n))
# df_all_greedy.append(pd.read_csv('lio/results/er%d/er4_2_greedy/log.csv'%n))

signal1= []
signal2= []



t = df_all_pure[0]['episode']

for df in df_all_greedy:
    signal1.append(df['A1_reward_total'] + df['A2_reward_total'] + df['A3_reward_total'] + df['A4_reward_total'])


for df in df_all_pure:
    signal2.append(df['A1_reward_total'] + df['A2_reward_total'] + df['A3_reward_total'] + df['A4_reward_total'])


mins1 = np.min(signal1, axis=0)
maxs1 = np.max(signal1, axis=0)
means1 = np.mean(signal1, axis=0)

mins2 = np.min(signal2, axis=0)
maxs2 = np.max(signal2, axis=0)
means2 = np.mean(signal2, axis=0)




plt.figure(figsize=(7, 4))
plt.fill_between(t, mins1, maxs1, alpha=0.3, color='tab:blue')
plt.plot(t, means1, color='tab:blue', label='Partial Communication', linewidth=1)

plt.fill_between(t, mins2, maxs2, alpha=0.3, color='tab:orange' )
plt.plot(t, means2, color='tab:orange', label='LIO', linewidth=1)


plt.grid(color = 'silver', linestyle = '--', linewidth = 0.3)

plt.xlabel('Episode',fontsize=18)
plt.ylabel('Total Rewards',fontsize=18)
plt.title('ER(4,2) ',fontsize=18)
plt.legend(title='Algorithm', title_fontsize = 14, loc='best')
plt.show()

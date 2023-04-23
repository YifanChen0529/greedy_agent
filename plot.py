import numpy as np
import matplotlib
import pandas as pd

# matplotlib.use('Qt4Cairo')
import matplotlib.pyplot as plt

# df = pd.read_csv('lio/results/er/er_n2_pg_cont/log.csv')
# df = pd.read_csv('lio/results/er/er_n2_lio/log.csv')
df =   pd.read_csv('lio/results/ipd/ipd_lio/log.csv')



# plt.figure(figsize=(12, 12))
# plt.plot(df['episode'], df['A1_reward_env'] + df['A2_reward_env'] , linestyle=':', marker='o',markersize=1 , color='red', label='Agent1', linewidth=1)
# plt.plot(df['episode'], df['A2_reward_env'], linestyle='-.', marker='o',markersize=1 , color='olive', label='Agent2', linewidth=1)
# # plt.plot(df['episode'], df['mission_status'], linestyle=':', marker='o',markersize=1 , color='orange', label='misssion status', linewidth=1)
# # plt.plot(df['episode'], df['A4_reward_total'], linestyle='-.', marker='o',markersize=1 , color='blue', label='Agent4', linewidth=1)
# plt.grid(color = 'silver', linestyle = '--', linewidth = 0.3)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('Agent Rewards')
# plt.legend(title='Agents', title_fontsize = 13, loc='best')
# plt.show()






# fig, ax1 = plt.subplots(figsize=(12, 12))
# ax1.plot(df['episode'], df['A1_reward_total'], linestyle=':', marker='o',markersize=1 , color='red', label='Agent1', linewidth=1)
# ax1.plot(df['episode'], df['A2_reward_total'], linestyle='-.', marker='o',markersize=1 , color='olive', label='Agent2', linewidth=1)
# # ax1.plot(df['episode'], df['A3_reward_total'], linestyle='-.', marker='o',markersize=1 , color='olive', label='Agent2', linewidth=1)
# # ax1.plot(df['episode'], df['A4_reward_total'], linestyle='-.', marker='o',markersize=1 , color='olive', label='Agent2', linewidth=1)
# ax1.set_xlabel('Episode')
# ax1.set_ylabel('Reward')
# # ax1.tick_params('y', colors='b')

# # Create the second plot with different y-axis
# ax2 = ax1.twinx()
# ax2.plot(df['episode'], df['A1_win_rate'], linestyle='-',markersize=1 , color='blue', linewidth=0.1)
# ax2.set_ylabel('Win_rate', color='blue')
# ax2.tick_params('y', colors='blue')

# plt.legend(title='Agents', title_fontsize = 13, loc='best')
# plt.title('Agent Rewards')
# plt.show()




# plt.figure(figsize=(12, 12))
# plt.plot(df['episode'], df['A1_reward_total'] + df['A2_reward_total'], linestyle='--', marker='o',markersize=1 , color='green', linewidth=1)
# plt.grid(color = 'silver', linestyle = '--', linewidth = 0.3)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('Total Rewards')
# plt.show()











# df_all_pure = []
# df_all_greedy = []
# for i in range(20):
#     df_all_pure.append(pd.read_csv('lio/results/er%d/er_n2_1T_pure_LIO/log.csv'%(i+1)))
#     df_all_greedy.append(pd.read_csv('lio/results/er%d/er_n2_1T_greedy_LIO/log.csv'%(i+1)))

# signal1= []
# signal2= []
# t = df_all_pure[0]['episode']
# for df in df_all_pure:
#     signal1.append(df['A1_reward_total'] + df['A2_reward_total'])

# for df in df_all_greedy:
#     signal2.append(df['A1_reward_total'] + df['A2_reward_total'])


# mins1 = np.min(signal1, axis=0)
# maxs1 = np.max(signal1, axis=0)
# means1 = np.mean(signal1, axis=0)

# mins2 = np.min(signal2, axis=0)
# maxs2 = np.max(signal2, axis=0)
# means2 = np.mean(signal2, axis=0)



# plt.figure(figsize=(7, 5))
# plt.fill_between(t, mins1, maxs1, alpha=0.3, color='tab:blue')
# plt.plot(t, means1, color='tab:blue', label='LIO', linewidth=1)

# plt.fill_between(t, mins2, maxs2, alpha=0.3, color='tab:orange' )
# plt.plot(t, means2, color='tab:orange', label='Partial Communication', linewidth=1)

# plt.grid(color = 'silver', linestyle = '--', linewidth = 0.3)

# plt.xlabel('Episode',fontsize=16)
# plt.ylabel('Total Agents Reward',fontsize=16)
# plt.title('ER(2,1) ',fontsize=16)
# plt.legend(title='Algorithm', title_fontsize = 13, loc='best')
# plt.show()



# df_all_pure = []
# df_all_greedy = []
# for i in range(20):
#     df_all_pure.append(pd.read_csv('lio/results/er%d/er_n2_1_pure_LIO/log.csv'%(i+1)))
#     df_all_greedy.append(pd.read_csv('lio/results/er%d/er_n2_1_greedy_LIO/log.csv'%(i+1)))

# signal1= []
# signal2= []
# t = df_all_pure[0]['episode']
# for df in df_all_pure:
#     signal1.append(df['A1_reward_total'] + df['A2_reward_total'] + df['A3_reward_total'] + df['A4_reward_total'])

# for df in df_all_greedy:
#     signal2.append(df['A1_reward_total'] + df['A2_reward_total'] + df['A3_reward_total'] + df['A4_reward_total'])


# mins1 = np.min(signal1, axis=0)
# maxs1 = np.max(signal1, axis=0)
# means1 = np.mean(signal1, axis=0)

# mins2 = np.min(signal2, axis=0)
# maxs2 = np.max(signal2, axis=0)
# means2 = np.mean(signal2, axis=0)

# print(len(signal2))


# plt.figure(figsize=(7, 5))
# plt.fill_between(t, mins1, maxs1, alpha=0.3, color='tab:blue')
# plt.plot(t, means1, color='tab:blue', label='LIO', linewidth=1)

# plt.fill_between(t, mins2, maxs2, alpha=0.3, color='tab:orange' )
# plt.plot(t, means2, color='tab:orange', label='Partial Communication', linewidth=1)

# plt.grid(color = 'silver', linestyle = '--', linewidth = 0.3)

# plt.xlabel('Episode',fontsize=16)
# plt.ylabel('Total Agents Reward',fontsize=16)
# plt.title('ER(4,2) ',fontsize=16)
# plt.legend(title='Algorithm', title_fontsize = 13, loc='best')
# plt.show()





# df_all_pure = []
# df_all_greedy = []
# for i in range(20):
#     df_all_pure.append(pd.read_csv('lio/results/er%d/er_n2_1T_pure_LIO/log.csv'%(i+1)))
#     df_all_greedy.append(pd.read_csv('lio/results/er%d/er_n2_1T_greedy_LIO/log.csv'%(i+1)))

# signal1= []
# signal2= []
# t = df_all_pure[0]['episode']
# # for df in df_all_pure:
# #     signal1.append(df['A1_reward_total'] )

# for df in df_all_greedy:
#     signal1.append(df['A1_reward_total'] )
#     signal2.append(df['A2_reward_total'])


# mins1 = np.min(signal1, axis=0)
# maxs1 = np.max(signal1, axis=0)
# means1 = np.mean(signal1, axis=0)

# mins2 = np.min(signal2, axis=0)
# maxs2 = np.max(signal2, axis=0)
# means2 = np.mean(signal2, axis=0)

# print(len(signal2))


# plt.figure(figsize=(7, 5))
# plt.fill_between(t, mins1, maxs1, alpha=0.3, color='tab:green')
# plt.plot(t, means1, color='tab:green', label='Adv agent', linewidth=1)

# plt.fill_between(t, mins2, maxs2, alpha=0.3, color='tab:orange' )
# plt.plot(t, means2, color='tab:orange', label='Cooperator', linewidth=1)

# plt.grid(color = 'silver', linestyle = '--', linewidth = 0.3)

# plt.xlabel('Episode',fontsize=16)
# plt.ylabel('Each Agent Reward',fontsize=16)
# plt.title('ER(2,1) ',fontsize=16)
# plt.legend(title='Algorithm', title_fontsize = 13, loc='best')
# plt.show()




# df_all_pure = []
# df_all_greedy = []
# for i in range(20):
#     df_all_pure.append(pd.read_csv('lio/results/er%d/er_n2_1_pure_LIO/log.csv'%(i+1)))
#     df_all_greedy.append(pd.read_csv('lio/results/er%d/er_n2_1_greedy_LIO/log.csv'%(i+1)))

# signal1= []
# signal2= []
# signal3= []
# signal4= []
# t = df_all_pure[0]['episode']
# # for df in df_all_pure:
# #     signal1.append(df['A1_reward_total'] )

# for df in df_all_greedy:
#     signal1.append(df['A1_reward_total'] )
#     signal2.append(df['A2_reward_total'])
#     signal3.append(df['A3_reward_total'])
#     signal4.append(df['A4_reward_total'])


# mins1 = np.min(signal1, axis=0)
# maxs1 = np.max(signal1, axis=0)
# means1 = np.mean(signal1, axis=0)

# mins2 = np.min(signal2, axis=0)
# maxs2 = np.max(signal2, axis=0)
# means2 = np.mean(signal2, axis=0)

# mins3 = np.min(signal3, axis=0)
# maxs3 = np.max(signal3, axis=0)
# means3 = np.mean(signal3, axis=0)

# mins4 = np.min(signal4, axis=0)
# maxs4 = np.max(signal4, axis=0)
# means4 = np.mean(signal4, axis=0)

# print(len(signal2))


# plt.figure(figsize=(7, 5))
# plt.fill_between(t, mins1, maxs1, alpha=0.3, color='tab:green')
# plt.plot(t, means1, color='tab:green', label='Adv agent', linewidth=1)

# plt.fill_between(t, mins2, maxs2, alpha=0.3, color='tab:orange' )
# plt.plot(t, means2, color='tab:orange', label='Cooperator', linewidth=1)

# plt.plot(t, means3, color='tab:orange', label='Cooperator', linewidth=1)
# plt.plot(t, means4, color='tab:orange', label='Cooperator', linewidth=1)

# plt.grid(color = 'silver', linestyle = '--', linewidth = 0.3)

# plt.xlabel('Episode',fontsize=16)
# plt.ylabel('Each Agent Reward',fontsize=16)
# plt.title('ER(4,2) ',fontsize=16)
# plt.legend(title='Algorithm', title_fontsize = 13, loc='best')
# plt.show()




# df_all_pure = []
# df_all_greedy = []
# for i in range(1,20):
#     df_all_pure.append(pd.read_csv('lio/results/ipd%d/ipd_greedy_lio/log.csv'%(i+1)))
#     df_all_greedy.append(pd.read_csv('lio/results/ipd%d/ipd_greedy_lio/log.csv'%(i+1)))

# signal1= []
# signal2= []
# t = df_all_pure[0]['episode']


# # for df in df_all_pure:
# #     signal1.append(df['A1_reward_env'])
# #     signal2.append(df['A2_reward_env'])

# for df in df_all_greedy:
#     signal1.append(df['A1_reward_env'] )
#     signal2.append(df['A2_reward_env'])


# mins1 = np.min(signal1, axis=0)
# maxs1 = np.max(signal1, axis=0)
# means1 = np.mean(signal1, axis=0)

# mins2 = np.min(signal2, axis=0)
# maxs2 = np.max(signal2, axis=0)
# means2 = np.mean(signal2, axis=0)



# plt.figure(figsize=(7, 5))
# plt.fill_between(t, mins1, maxs1, alpha=0.3, color='tab:green')
# plt.plot(t, means1, color='tab:green', label='Adv agent', linewidth=1)

# plt.fill_between(t, mins2, maxs2, alpha=0.3, color='tab:orange' )
# plt.plot(t, means2, color='tab:orange', label='Cooperator', linewidth=1)

# plt.grid(color = 'silver', linestyle = '--', linewidth = 0.3)

# plt.xlabel('Episode',fontsize=16)
# plt.ylabel('Each Agent env Reward',fontsize=16)
# plt.title('IPD ',fontsize=16)
# plt.legend(title='Agent', title_fontsize = 13, loc='best')
# plt.show()





df_all_pure = []
df_all_greedy = []
for i in range(20):
    df_all_pure.append(pd.read_csv('lio/results/er%d/er_n2_1_adv_a2d_LIO/log.csv'%(i+1)))
    df_all_greedy.append(pd.read_csv('lio/results/er%d/er_n2_1T_adv_a2d_LIO/log.csv'%(i+1)))

signal1= []
signal2= []

signal3= []
signal4= []

t = df_all_pure[0]['episode']
for df in df_all_pure:
    signal1.append(df['A1_reward_total'])
    signal2.append(df['A2_reward_total'])
    signal3.append(df['A3_reward_total'])
    signal4.append(df['A4_reward_total'])

# for df in df_all_greedy:
#     signal1.append(df['A1_reward_total'])
#     signal2.append(df['A2_reward_total'])


mins1 = np.min(signal1, axis=0)
maxs1 = np.max(signal1, axis=0)
means1 = np.mean(signal1, axis=0)

mins2 = np.min(signal2, axis=0)
maxs2 = np.max(signal2, axis=0)
means2 = np.mean(signal2, axis=0)


mins3 = np.min(signal3, axis=0)
maxs3 = np.max(signal3, axis=0)
means3 = np.mean(signal3, axis=0)

mins4 = np.min(signal4, axis=0)
maxs4 = np.max(signal4, axis=0)
means4 = np.mean(signal4, axis=0)

print(len(signal2))


plt.figure(figsize=(7, 5))
plt.fill_between(t, mins1, maxs1, alpha=0.3, color='tab:green')
plt.plot(t, means1, color='tab:green', label='Adv agent', linewidth=1)

plt.fill_between(t, mins2, maxs2, alpha=0.3, color='tab:orange' )
plt.plot(t, means2, color='tab:orange', label='Cooperator', linewidth=1)

plt.fill_between(t, mins3, maxs3, alpha=0.3, color='tab:orange' )
plt.plot(t, means3, color='tab:orange', label='Cooperator', linewidth=1)
plt.fill_between(t, mins4, maxs4, alpha=0.3, color='tab:orange' )
plt.plot(t, means4, color='tab:orange', label='Cooperator', linewidth=1)

plt.grid(color = 'silver', linestyle = '--', linewidth = 0.3)

plt.xlabel('Episode',fontsize=16)
plt.ylabel('Each Agent Reward',fontsize=16)
plt.title('ER(2,1) ',fontsize=16)
plt.legend(title='Agents', title_fontsize = 13, loc='best')
plt.show()


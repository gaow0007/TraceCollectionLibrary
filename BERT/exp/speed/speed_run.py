from numpy import histogram


def repackage(num_node, num_replicas): 
    if num_node > 8: 
        num_node //= 2 
    per_replicas = num_replicas // num_node 
    placement_list = [per_replicas for _ in range(num_node)]
    remaining = num_replicas - sum(placement_list)
    for i in range(remaining): 
        placement_list[i] += 1
    return placement_list

def scability_loader(): 
    for num_node in [6, 8, 12, 16]: 
        for num_replicas in range(num_node, num_node * 4+4, 4): 
            yield repackage(num_node, num_replicas)

# scability_list = list() 
# for scability in scability_loader(): 
#     if scability not in scability_list: 
#         print(scability)
#         scability_list.append(scability)
# # print(scability_loader())
# exit(0)


def info_loader(): 
    for a1 in range(5): 
        for a2 in range(5): 
            for a3 in range(5): 
                for a4 in range(5): 
                    yield [a1, a2, a3, a4]

def isLegal(placement): 
    for i in range(3): 
        if placement[i] < placement[i+1]: 
            return False
    return True 

arch_list = ['WikiText2'] # , 'WikiText103'] 
node_list = ['79', '75', '67', '65'] 
# node_list = ['34', '39', '30']
history_list = list() 

print('set -e')
cnt = 0
for placement in info_loader(): 
    if isLegal(placement): 
        while 0 in placement: 
            placement.remove(0) 
        if len(placement) == 0: continue 
        if len(placement) < 3: continue 

        placement_str = '-'.join([str(pm) for pm in placement])
        if placement_str in history_list: 
            continue 

        history_list.append(placement_str)

        for arch in arch_list: 
            # bash exp/slurm_run_speed.sh 1 $arch 1 $node 
            node_str = "'{}'".format(' '.join(node_list[:len(placement)]))
            placement_str = [str(pm) for pm in placement]
            cmd = 'bash exp/speed/slurm_run_auto_speed_{}.sh {} {} {} {}'.format(cnt%2, sum(placement), arch, "'{}'".format(' '.join(placement_str)), node_str)
            print('''echo "{}" '''.format(cmd))
            if cnt % 2 == 0: 
                # print(cmd+'&') 
                print(cmd)
            else: 
                print(cmd)
            cnt += 1
            
        # break 
# python exp/speed_run.py > exp/placement_speed.sh 
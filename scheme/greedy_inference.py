import torch

fill_empty = True
syntax = True

# Algorithm 1: Greedy Inference
def loop_version_from_tag_table_to_triplets(tag_table, adj_table, id2senti, version='3D'):
    
    raw_table_id = torch.tensor(tag_table)
    adj_table_id = torch.tensor(adj_table)
    
    # line 1 to line 4  (get aspect/opinion/sentiment snippet)
    if version == '1D': # {N, NEG, NEU, POS, O, A}
        if_aspect = (raw_table_id == 5) > 0
        if_opinion = (raw_table_id == 4) > 0
        if_triplet = raw_table_id * ((raw_table_id > 0) * (raw_table_id < 4)) 
    else: # 2D: {N,O,A} - {N, NEG, NEU, POS}  #3D: {N,A} - {N,O} - {N, NEG, NEU, POS}
        if_aspect = (raw_table_id & torch.tensor(8)) > 0
        if_opinion = (raw_table_id & torch.tensor(4)) > 0
        if_triplet = (raw_table_id & torch.tensor(3))
    if_adj = (adj_table_id == 1) > 0
    
    m = if_triplet.nonzero()
    senti = if_triplet[m[:,0],m[:,1]].unsqueeze(dim=-1)
    candidate_triplets = torch.cat([m,senti,m.sum(dim=-1,keepdim=True)],dim=-1).tolist()
    candidate_triplets.sort(key = lambda x:(x[-1],x[0]))
    
    
    valid_triplets = []
    
    valid_triplets_set = set([])
    
    
    # line 5 to line 24 (look into every sentiment snippet)
    for r_begin, c_end, p, _ in candidate_triplets:
        
        #####################################################################################################
        # CASE-1: aspect-opinion        
        aspect_candidates = guarantee_list((if_aspect[r_begin, r_begin:(c_end+1)].nonzero().squeeze()+r_begin).tolist()) # line 7
        opinion_candidates = guarantee_list((if_opinion[r_begin:(c_end+1),c_end].nonzero().squeeze()+r_begin).tolist())  # line 8

        if len(aspect_candidates) and len(opinion_candidates):  # line 9
            if syntax and adj_table:
                #print(if_adj)
                #print(if_aspect)
                aspect_certified = guarantee_list(
                    (if_adj[r_begin, r_begin:(c_end + 1)].nonzero().squeeze() + r_begin).tolist())
                opinion_certified = guarantee_list(
                    (if_adj[r_begin:(c_end + 1), c_end].nonzero().squeeze() + r_begin).tolist())
                select_aspect_c = find_common_element(aspect_candidates, aspect_certified, True)
                select_opinion_r = find_common_element(opinion_candidates, opinion_certified, False)
            else:
                # print(opinion_candidates)
                select_aspect_c = -1
                select_opinion_r = 0
            #select_aspect_c = -1 if (len(aspect_candidates) == 1 or aspect_candidates[-1] != c_end) else -2     # line 10
            #select_opinion_r = 0 if (len(opinion_candidates) == 1 or opinion_candidates[0] != r_begin) else 1   # line 11
            
            # line 12
            a_ = [r_begin, aspect_candidates[select_aspect_c]]  
            o_ = [opinion_candidates[select_opinion_r], c_end] 
            s_ = id2senti[p] #id2label[p]
            
            # line 13
            if str((a_,o_,s_)) not in valid_triplets_set:
                valid_triplets.append((a_,o_,s_))
                valid_triplets_set.add(str((a_,o_,s_)))
            
            
        #####################################################################################################    
        # CASE-2: opinion-aspect
        opinion_candidates = guarantee_list((if_opinion[r_begin, r_begin:(c_end+1)].nonzero().squeeze()+r_begin).tolist())   # line 16
        aspect_candidates = guarantee_list((if_aspect[r_begin:(c_end+1),c_end].nonzero().squeeze()+r_begin).tolist())        # line 17

        if len(aspect_candidates) and len(opinion_candidates):  # line 18
            if syntax and adj_table:
                #print(if_adj)
                #print(if_aspect)
                opinion_certified = guarantee_list(
                    (if_adj[r_begin, r_begin:(c_end + 1)].nonzero().squeeze() + r_begin).tolist())
                aspect_certified = guarantee_list(
                    (if_adj[r_begin:(c_end + 1), c_end].nonzero().squeeze() + r_begin).tolist())
                select_opinion_c = find_common_element(opinion_candidates, opinion_certified, True)
                select_aspect_r = find_common_element(aspect_candidates, aspect_certified, False)
            else:
                select_opinion_c = -1
                select_aspect_r = 0
            #select_opinion_c = -1 if (len(opinion_candidates) == 1 or opinion_candidates[-1] != c_end) else -2 # line 19
            #select_aspect_r = 0 if (len(aspect_candidates) == 1 or aspect_candidates[0] != r_begin) else 1     # line 20
            
            # line 21
            o_ = [r_begin, opinion_candidates[select_opinion_c]]
            a_ = [aspect_candidates[select_aspect_r], c_end]
            s_ = id2senti[p] #id2label[p]
            
            # line 22
            if str((a_,o_,s_)) not in valid_triplets_set:
                valid_triplets.append((a_,o_,s_))
                valid_triplets_set.add(str((a_,o_,s_)))

        if fill_empty:
            # Add a default triplet if valid_triplets is empty
            if not valid_triplets and torch.sum(if_aspect) > 0 and torch.sum(if_opinion) > 0:
                # Take first aspect and first opinion from non-zero positions
                aspect_candidate = ensure_nested_list(if_aspect.nonzero().squeeze().tolist())[0]  # Take first non-zero aspect
                opinion_candidate = ensure_nested_list(if_opinion.nonzero().squeeze().tolist())[0]  # Take first non-zero opinion
                if len(candidate_triplets) > 0:
                    sentiment = id2senti[candidate_triplets[0][2]]  # Get sentiment from id2senti
                else:
                    sentiment = 'POS'
                valid_triplets.append((aspect_candidate, opinion_candidate, sentiment))  # Add the default triplet



    return {
        'aspects': if_aspect.nonzero().squeeze().tolist(), # for ATE
        'opinions': if_opinion.nonzero().squeeze().tolist(), # for OTE
        'triplets': sorted(valid_triplets, key=lambda x:(x[0][0],x[0][-1],x[1][0],x[1][-1])) # line 25
    }

def guarantee_list(l):
    if type(l) != list:
        l = [l]
    return l

def ensure_nested_list(input_list):
    # 判断输入是否为二层嵌套列表
    if isinstance(input_list, list) and all(isinstance(i, list) for i in input_list):
        return input_list
    else:
        return [input_list]


def find_common_element(a, b, reverse=False):
    b_set = set(b)  # 将列表 b 转换为集合，提高查找效率
    last_element = -1 if reverse else 0  # 记录最后查找的元素
    if reverse:
        # 从后向前遍历
        for i in range(len(a) - 1, -1, -1):
            if a[i] in b_set:
                return i  # 找到匹配元素则返回
            last_element = i
    else:
        # 从前向后遍历
        for i in range(len(a)):
            if a[i] in b_set:
                return i  # 找到匹配元素则返回
            last_element = i
    return last_element  # 如果未找到，返回最后一个查找的元素
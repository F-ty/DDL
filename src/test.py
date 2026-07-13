import numpy as np
import json
import torch
from tqdm import tqdm as tqdm
import torch.nn.functional as F

def get_sample(params, model, testset, category):
    model.eval()
    (test_queries, test_targets, name) = (testset.test_queries, testset.test_targets, category)
    #                                               dress  shirt  toptee  shoes
    # print("test_queries: ", len(test_queries))  # 5985   5988   6027    1761
    # print("test_targets: ", len(test_targets))  # 7757   8988   8695    4658

    # print(test_targets[0])
    ###
    # shoes
    # source_img_id, source_img_data
    # target_img_id, target_img_data
    
    with torch.no_grad():
        all_queries = []
        all_imgs = []
        if test_queries:
            visual_query = []
            textual_query = []
            for t in tqdm(test_queries, disable=False if params.local_rank == 0 else True):
                visual_query += [t['visual_query']]
                textual_query += [t['textual_query']]
                if len(visual_query) >= params.batch_size or t is test_queries[-1]:
                    visual_query = torch.stack(visual_query).float().cuda() # 5985 * 3 * 224 * 224
                    # print("visual_query: ", visual_query.shape)
                    f, _, _, _, _, _ = model.extract_query(textual_query, visual_query) # 5984 * 512
                    # print("f: ", f.shape)
                    f = f.data.cpu().numpy()
                    all_queries += [f]

                    visual_query = []
                    textual_query = []
            
            all_queries = np.concatenate(all_queries)

            imgs = []
            logits = []
            for t in tqdm(test_targets, disable=False if params.local_rank == 0 else True):
                imgs += [t['target_img_data']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs = model.extract_target(imgs).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)

    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])  
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    sims = all_queries.dot(all_imgs.T)

    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
        
    # print(len(data1))

    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
    
    # shoes
    path1 = "/mnt/disk/fty_cxsj/DQU-CIR/src/img_all.json"
    with open(path1, "r") as file:
        data1 = json.load(file)
    k = 10
    print("test_queries: ", len(test_queries))
    print("test_targets:", len(test_targets))
    print("all_queries: ", all_queries.shape)
    print("all_imgs: ", all_imgs.shape)
    print("nn_result: ", len(nn_result))
    for i, nns in enumerate(nn_result):
        q = test_queries[i]
        print("src_id: ", data1[q["source_img_id"]])
        print("tar_id: ", data1[q["target_img_id"]])
        print("textual_query: ", q["textual_query"])
        for nn in nns[:k]:
            p = test_targets[nn]
            print(data1[p['target_img_id']])
        # print(test_queries[i])
        # print(nns)
        # print()
        # temp_dict = {}
            # print(test_queries[i])
        # temp_dict["target"] = test_queries[i]["target_img_path"]
        # lst = []
            # print("{}/{}".format(str(i), ))
            # print("{}--------------".format(i))
            # print(nns[:k])
        # if i != nns[0]:
        #         print('src: ', test_queries[i]['source_img_path'])
        #         print('mod: ', test_queries[i]['textual_query'])
        #         print("tar: ", test_queries[i]["target_img_path"])
        #         for nn in nns[:k]:
        #             print(test_queries[nn]["target_img_path"])
        #         print("\n")
                # cnt+=1
        # print("cnt:{}".format(cnt))
            # for nn in nns[:k]:
            #     path = test_queries[nn]["target_img_path"]
            #     print("{}/{}".format(str(i), str(nn)))
            #     # if path != train_queries[i]["target_img_path"] and path not in lst:
            #     lst.append(path)
            #     if len(lst) == k:
            #         break
            # temp_dict["sims"] = lst
            # sims_lst.append(temp_dict)
        
        # print("sims_lst: ", len(sims_lst)) # 5985
        # with open("{}_test_sim.json".format(params.dataset), "w") as file:
        #     json.dump(sims_lst, file)

def eval_train_sim(params, model, testset, category):
    model.eval()
    (train_queries, train_targets, name) = (testset.train_queries, testset.train_targets, category)
    #                                               dress  shirt  toptee
    print("train_queries: ", len(train_queries))  # 5985   5988   6027
    print("train_targets: ", len(train_targets))  # 7757   8988   8695

    with torch.no_grad():
        visual_query = []
        for t in tqdm(train_queries, disable=False if params.local_rank == 0 else True):
            visual_query += [t['target_img_data']]
        visual_query = torch.stack(visual_query).float().cuda() # 5985 * 3 * 224 * 224
        print("visual_query: ", visual_query.shape)
        f = model.extract_target(visual_query).data.cpu().numpy() # 5984 * 512
        print("f: ", f.shape)
    
        for i in range(f.shape[0]):
            f[i, :] /= np.linalg.norm(f[i, :])

        sims = f.dot(f.T) # 5984 * 5985

        # np.save("sims.npy", sims)

        # train_targets_id = []
        # for i in train_targets:
        #     train_targets_id.append(i['target_img_id'])
        
        nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

        k = 20
        sims_lst = []
        for i, nns in enumerate(nn_result):
            temp_dict = {}
            temp_dict["target"] = train_queries[i]["target_img_path"]
            lst = []
            for nn in nns:
                path = train_queries[nn]["target_img_path"]
                if path != train_queries[i]["target_img_path"] and path not in lst:
                    lst.append(path)
                if len(lst) == k:
                    break
            temp_dict["sims"] = lst
            sims_lst.append(temp_dict)
        
        print("sims_lst: ", len(sims_lst)) # 5985
        # with open("{}_train_sim.json".format(params.dataset), "w") as file:
        #     json.dump(sims_lst, file)

        # nn = nn_result[0][:10]
        # # print("nn: ", nn)
        # print("nn: ", train_queries[0]["target_img_path"])
        # for i in nn:
        #     print(train_queries[i]["target_img_path"])

        # for i, nns in enumerate(nn_result):
            

        # out = []
        # for k in [1, 10, 50]:
        #     r = 0.0
        #     for i, nns in enumerate(nn_result):
        #         if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
        #             r += 1
        #     r = 100 * r / len(nn_result)
        #     out += [('{}_r{}'.format(name, k), r)]

        # return out
        

def test(params, model, testset, category):
    model.eval()

    (test_queries, test_targets, name) = (testset.test_queries, testset.test_targets, category)
    # test_queries: list
    #    visual_query (tensor), source_img_path, source_img_id, textual_query (str)
    #    target_img_id, target_img_data (tensor), target_img_path

    # test_targets: list
    #    target_img_id, target_img_data (tensor), target_img_path

    with torch.no_grad():
        all_queries = []
        all_imgs = []
        if test_queries:
            visual_query = []
            textual_query = []
            for t in tqdm(test_queries, disable=False if params.local_rank == 0 else True):
                visual_query += [t['visual_query']]
                textual_query += [t['textual_query']]
                if len(visual_query) >= params.batch_size or t is test_queries[-1]:

                    visual_query = torch.stack(visual_query).float().cuda()
                    # print("visual_query: ", visual_query.shape)
                    # print("textual_query: ", len(textual_query))
                    # f = model.extract_query(textual_query, visual_query)
                    f, _, _, _, _, _ = model.extract_query(textual_query, visual_query)

                    f = f.data.cpu().numpy()
                    all_queries += [f]

                    visual_query = []
                    textual_query = []
            
            # print("all_queries: ", len(all_queries))
            all_queries = np.concatenate(all_queries)
            # print("all_queries: ", all_queries.shape)

            # compute all image features
            imgs = []
            logits = []
            for t in tqdm(test_targets, disable=False if params.local_rank == 0 else True):
                imgs += [t['target_img_data']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs = model.extract_target(imgs).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
    
    
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    
    test_targets_id = []
    # test_dict = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    #     test_dict.append(i['target_img_path'])
    # temp = {"id":test_dict}
    # import json
    # with open('dress_test_id.json', 'w') as f:
    #     json.dump(temp, f)


    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

    # k = 10
    # for i, nns in enumerate(nn_result):
    #     temp = nns[:k]
    #     test_targets_id
    
    # mrr = 0
    # for i, nns in enumerate(nn_result):
    #     if test_targets_id.index(test_queries[i]['target_img_id']) in nns:
    #         # print(nns)
    #         # print(test_targets_id.index(test_queries[i]['target_img_id']))
    #         idx = np.where(nns==test_targets_id.index(test_queries[i]['target_img_id']))[0][0] + 1
    #         mrr += 1 / idx
    # print("mrr: ", mrr / len(nn_result))

    # k = 5
    # ndcg_scores = []
    # for i, nns in enumerate(nn_result):
        
    #     top_k_sims = sims[i][nns[:k]]
        
    #     relevance_scores = []
    #     for sim in top_k_sims:
    #         if sim >= 0.6:
    #             relevance_scores.append(3)
    #         elif sim >= 0.3:
    #             relevance_scores.append(2)
    #         elif sim >= 0.0:
    #             relevance_scores.append(1)
    #         else:
    #             relevance_scores.append(0)
                
    #     dcg = 0
    #     for j, score in enumerate(relevance_scores):
    #         dcg += score / np.log2(j + 2)
            
    #     idcg = 0
    #     ideal_scores = [3] * k
    #     for j, score in enumerate(ideal_scores):
    #         idcg += score / np.log2(j + 2)

    #     if idcg > 0:
    #         ndcg = dcg / idcg
    #     else:
    #         ndcg = 0
            
    #     ndcg_scores.append(ndcg)
        
    # mean_ndcg = np.mean(ndcg_scores)
    # print("NDCG@5: {:.4f}".format(mean_ndcg))
    
    out = []
    for k in [1, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        out += [('{}_r{}'.format(name, k), r)]


    return out


def test_fashion200k_dataset(params, model, testset):
    """Tests a model over the given testset."""
    
    model.eval()
    test_queries = testset.get_test_queries()
    # print("test_queries: ", len(test_queries)) # 33480
    # print("testset.imgs: ", len(testset.imgs)) # 29789
    with torch.no_grad():
        all_imgs = []
        all_captions = []
        all_queries = []
        all_target_captions = []
        if test_queries:
            # compute test query features
            imgs = []
     
            visual_query = []
            textual_query = []
            for t in test_queries:
                visual_query += [testset.get_written_img(t['source_img_id'], t['target_word'])]
                textual_query += [t['source_caption'] + ', but ' + t['mod']['str']]

                if len(visual_query) >= params.batch_size or t is test_queries[-1]:
                    visual_query = torch.stack(visual_query).float().cuda()
                    f, _, _, _, _, _ = model.extract_query(textual_query, visual_query)
                    f = f.data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    visual_query = []
                    textual_query = []
            all_queries = np.concatenate(all_queries)
            all_target_captions = [t['target_caption'] for t in test_queries]

            # compute all image features
            imgs = []
            for i in range(len(testset.imgs)):
                imgs += [testset.get_img(i)]
                if len(imgs) >= params.batch_size or i == len(testset.imgs) - 1:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs = model.extract_target(imgs).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)
            all_captions = [img['captions'][0] for img in testset.imgs]

        # feature normalization
        for i in range(all_queries.shape[0]):
            all_queries[i, :] /= np.linalg.norm(all_queries[i, :])

        for i in range(all_imgs.shape[0]):
            all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

        # match test queries to target images, get nearest neighbors
        sims = all_queries.dot(all_imgs.T)
        if test_queries:
            for i, t in enumerate(test_queries):
                sims[i, t['source_img_id']] = -10e10  
        nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
        # print("nn_result: ", len(nn_result)) # 33480
        # print("nn_result[0]: ", nn_result[0].shape) # 110
        # print(nn_result[0])
        # compute recalls
        out = []
        tmp = nn_result
        nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
        # print("nn_result: ", len(nn_result)) # 33480

        # test_queries: 33480
        # print(len(nn_result)) 33480
        
        # k = 5
        # for i, (nns, id) in enumerate(zip(nn_result, tmp)):
        #     q = test_queries[i]
        # #     print("all_tar: ", all_target_captions[i])
        #     if all_target_captions[i] in nns[:k]:
        #         print(q)
        #         print(id[:k])
        #         for j, nn in enumerate(nns[:k]):
        #             print(nn)
        #             print(testset.imgs[id[j]]['file_path'])
        
        mrr = 0
        for i, nns in enumerate(nn_result):
            if all_target_captions[i] in nns:
                # print(nns)
                # print(all_target_captions[i])
                idx = nns.index(all_target_captions[i]) + 1
                # idx = np.where(nns==test_targets_id.index(test_queries[i]['target_img_id']))[0][0] + 1
                mrr += 1 / idx
        print("mrr: ", mrr / len(nn_result))
        
        for k in [1, 10, 50]:
            r = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i] in nns[:k]:
                    r += 1
            r /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_composition', r)]

        # 计算NDCG@5
        k = 5
        ndcg_scores = []
        for i, nns in enumerate(nn_result):
            # 获取top-k结果的相似度
            top_k_sims = sims[i][nns[:k]]
            
            # 计算相关性得分
            relevance_scores = []
            for sim in top_k_sims:
                if sim >= 0.6:
                    relevance_scores.append(3)
                elif sim >= 0.3:
                    relevance_scores.append(2)
                elif sim >= 0.0:
                    relevance_scores.append(1)
                else:
                    relevance_scores.append(0)
                    
            # 计算DCG
            dcg = 0
            for j, score in enumerate(relevance_scores):
                dcg += score / np.log2(j + 2)  # j+2是因为log的底数从2开始
                
            # 计算IDCG (假设最佳排序是所有结果都是最高分3)
            idcg = 0
            ideal_scores = [3] * k
            for j, score in enumerate(ideal_scores):
                idcg += score / np.log2(j + 2)
                
            # 计算NDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0
                
            ndcg_scores.append(ndcg)
            
        # 计算平均NDCG@5
        mean_ndcg = np.mean(ndcg_scores)
        print("NDCG@5: {:.4f}".format(mean_ndcg))

        return out
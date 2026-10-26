"""
数据加载与划分
关键修复：
1. 长度过滤从 >= 10 改为 >= 1（保留所有非空文本）
2. 严格按来源分层抽样
3. 数据增强仅在训练集内进行
"""

import os
import random
import numpy as np
from collections import defaultdict, Counter


def load_raw_data(base_path):
    """加载原始数据"""
    texts, labels, sources = [], [], []
    
    level_mapping = {
        '初级（1-3级）': {'1级': 0, '2级': 1, '3级': 2},
        '中级（4-6级）': {'4级': 3, '5级': 4, '6级': 5},
        '高级（7-9级）': {'7-9级': 6}
    }
    
    for main_level, sub_levels in level_mapping.items():
        main_path = os.path.join(base_path, main_level)
        if not os.path.exists(main_path):
            print(f"[Warning] Path not found: {main_path}")
            continue
        
        for sub_level, label in sub_levels.items():
            if main_level == '高级（7-9级）':
                for item in os.listdir(main_path):
                    item_path = os.path.join(main_path, item)
                    if os.path.isdir(item_path):
                        source = f"{main_level}/{item}"
                        _load_from_dir(item_path, texts, labels, sources, label, source)
            else:
                sub_path = os.path.join(main_path, sub_level)
                if os.path.exists(sub_path):
                    source = f"{main_level}/{sub_level}"
                    _load_from_dir(sub_path, texts, labels, sources, label, source)
    
    print(f"[Info] Loaded {len(texts)} texts from {base_path}")
    print(f"[Info] Label distribution: {dict(Counter(labels))}")
    print(f"[Info] Number of unique sources: {len(set(sources))}")
    
    return texts, labels, sources


def _load_from_dir(directory, texts, labels, sources, label, source):
    """递归加载目录下所有txt文件"""
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            _load_from_dir(item_path, texts, labels, sources, label, source)
        elif item.endswith('.txt'):
            try:
                with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                text = text.replace('\n', ' ').replace('\r', ' ').strip()
                # 放宽过滤：仅跳过完全空白的文件
                if len(text) >= 1:
                    texts.append(text)
                    labels.append(label)
                    sources.append(source)
            except Exception:
                continue


def split_by_source(texts, labels, sources, seed, ratios=(0.8, 0.1, 0.1)):
    """严格按来源分层抽样"""
    random.seed(seed)
    np.random.seed(seed)
    
    groups = defaultdict(list)
    for i, (s, l) in enumerate(zip(sources, labels)):
        groups[(s, l)].append(i)
    
    train_idx, val_idx, test_idx = [], [], []
    
    for key, indices in groups.items():
        random.shuffle(indices)
        n = len(indices)
        
        if n < 3:
            train_idx.extend(indices)
            continue
        
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1])) if n >= 5 else 0
        
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])
    
    print(f"[Info] Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"[Info] Total: {len(train_idx) + len(val_idx) + len(test_idx)}")
    
    return (
        [texts[i] for i in train_idx], [labels[i] for i in train_idx],
        [texts[i] for i in val_idx], [labels[i] for i in val_idx],
        [texts[i] for i in test_idx], [labels[i] for i in test_idx]
    )


def augment_training_only(train_texts, train_labels, target=800, seed=42):
    """数据增强（仅在训练集内进行）"""
    random.seed(seed)
    np.random.seed(seed)
    
    by_class = defaultdict(list)
    for t, l in zip(train_texts, train_labels):
        by_class[l].append(t)
    
    aug_texts, aug_labels = [], []
    
    for label in range(7):
        items = by_class.get(label, [])
        if not items:
            continue
        
        aug_texts.extend(items)
        aug_labels.extend([label] * len(items))
        
        if len(items) < target:
            needed = target - len(items)
            for _ in range(needed):
                src = random.choice(items)
                aug_texts.append(_char_augment(src, seed=random.randint(0, 99999)))
                aug_labels.append(label)
    
    combined = list(zip(aug_texts, aug_labels))
    random.shuffle(combined)
    aug_texts, aug_labels = zip(*combined)
    
    print(f"[Info] After augmentation: {len(aug_texts)} samples")
    print(f"[Info] Distribution: {dict(Counter(aug_labels))}")
    
    return list(aug_texts), list(aug_labels)


def _char_augment(text, seed):
    """字符级增强"""
    random.seed(seed)
    chars = list(text)
    
    if len(chars) < 5:
        return text
    
    op = random.choice(['delete', 'swap', 'replace'])
    
    if op == 'delete' and len(chars) > 3:
        idx = random.randint(0, len(chars) - 1)
        del chars[idx]
    elif op == 'swap' and len(chars) > 2:
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == 'replace':
        i = random.randint(0, len(chars) - 1)
        chars[i] = random.choice('的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实日军者意无力它与长把机十民第公此已工使情明性知全三又关点正业外将两高间由问很最重并物手应战向头文体政美相见被利什二等产或新己制身果加西斯月话合回特代内信表化老给世位次度门任常先海通教儿原东声提立及比员解水名真论处走义各入几口认条平系气题活尔更别打女变四神总何电数安少报才结反受目太量再感建务做接必场件计管期市直德资命山金指克许统区保至队形社便空决治展马科司五基眼书非则听白却界达光放强即像难且权思王象完设式色路记南品住告类求据程北边死张该交规万取拉格望觉术领共确传师观清今切院让识候带导争运笑飞风步改收根干造言联持组每济车亲极林服快办议往元英士证近失转夫令准布始怎呢存未远叫台单影具罗字爱击流备兵连调深商算质团集百需价花党华城石级整府离况亚请技际约示复病息究线似官火断精满支视消越器容照须九增研写称企八功吗包片史委乎查轻易早曾除农找装广显吧阿李标谈吃图念六引历首医局突专费号尽另周较注语仅考落青随选列武红响虽推势参希古众构房半节土投某案黑维革划敌致陈律足态护七兴派孩验责营星够章音跟志底站严巴例防族供效续施留讲型料终答紧黄绝奇察母京段依批群项故按河米围江织害斗双境客纪采举杀攻父苏密低朝友诉止细愿千值仍男钱破网热助倒育属坐帝限船脸职速刻乐否刚威毛状率甚独球般普怕弹校苦创假久错承印晚兰试股拿脑预谁益阳若哪微尼继送急血惊伤素药适波夜省初喜卫源食险待述陆习置居劳财环排福纳欢雷警获模充负云停木游龙树疑层冷洲冲射略范竟句室异激汉村哈策演简卡罪判担州静退既衣您宗积余痛检差富灵协角占配征修皮挥胜降阶审沉坚善妈刘读啊超免压银买皇养伊怀执副乱抗犯追帮宣佛岁航优怪香著田铁控税左右份穿艺背阵草脚概恶块顿敢守酒岛托央户烈洋哥索胡款靠评版宝座释景顾弟登货互付伯慢欧换闻危忙核暗姐介坏讨丽良序升监临亮露永呼味野架域沙掉括舰鱼杂误湾吉减编楚肯测败屋跑梦散温困剑渐封救贵枪缺楼县尚毫移娘朋画班智亦耳恩短掌恐遗固席松秘谢鲁遇康虑幸均销钟诗藏赶剧票损忽巨炮旧端探湖录叶春乡附吸予礼港雨呀板庭妇归睛饭额含顺输摇招婚脱补谓督毒油疗旅泽材灭逐莫笔亡鲜词圣择寻厂睡博勒烟授诺伦岸奥唐卖俄炸载洛健堂旁宫喝借君禁阴园谋宋避抓荣姑孙逃牙束跳顶玉镇雪午练迫爷篇肉嘴馆遍凡础洞卷坦牛宁纸诸训私庄祖丝翻暴森塔默握戏隐熟骨访弱蒙歌店鬼软典欲萨伙遭盘爸扩盖弄雄稳忘亿刺拥徒姆杨齐赛趣曲刀床迎冰虚玩析窗醒妻透购替塞努休虎扬途侵刑绿兄迅套贸毕唯谷轮库迹尤竞街促延震弃甲伟麻川申缓潜闪售灯针哲络抵朱埃抱鼓植纯夏忍页杰筑折郑贝尊吴秀混臣雅镇振染盛怒舞圆搞狂措姓残秋培迷诚宽宇猛摆梅毁伸摩盟末乃悲拍丁赵硬麦蒋操耶阻订曹仇奉申尚袁赫桑屡斥豪阵兼劝敬辞赞偷舍刀循颂菲沿悠凯赢洞秦诞琦谋襄洪洗汁悬牧催箭肃穆秋莎绳捷奴逊乃梯苛奋涨潮贾惯喉')
    
    return ''.join(chars)


def build_vocab(train_texts, vocab_size=60000):
    """从训练集构建词汇表"""
    counter = Counter()
    for t in train_texts:
        counter.update(list(t))
    
    most_common = counter.most_common(vocab_size - 2)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (c, _) in enumerate(most_common, start=2):
        vocab[c] = i
    
    print(f"[Info] Vocabulary size: {len(vocab)}")
    return vocab
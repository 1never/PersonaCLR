import re
import MeCab
import ipadic

tagger = MeCab.Tagger(f'-O wakati {ipadic.MECAB_ARGS}')

def normalize(text):
    text = text.replace("「", "").replace("」", "").replace("『", "").replace("』", "").replace("（", "").replace("）", "").replace("　", " ")
    text = text.replace("(", "").replace(")", "").replace("?", "？").replace("!", "！").replace("――", "").replace("!?", "！？").replace("～", "ー")
    text = re.sub(r"・・+", "…", text)
    text = re.sub(r"…+", "…", text)
    text = re.sub(r"^…", "", text)
    text = re.sub(r"…$", "", text)
    text = re.sub(r"ー+", "ー", text)
    text = re.sub(r"――+", "", text)
    text = re.sub(r"！+", "！", text)
    text = re.sub(r"？+", "？", text)
    text = text.replace("…？", "？").replace("…！", "！").replace("…、", "、").replace("…。", "。")
    return text

def word_count(text):
    text = normalize(text)
    return len(tagger.parse(text).split())

def parse(text):
    return tagger.parse(text)


def get_enames():
    return ["rimuru", "veldora", "rudeus", "roxy", "sylphy", "paul", "zenith", "subaru", "myne", "tuuli", "effa", "leon", "olivia", "angelica", "luxion", "catarina", "keith"]

def get_display_enames():
    return ["Rimuru", "Veldora", "Rudeus", "Roxy", "Sylphiette", "Paul", "Zenith", "Subaru", "Myne", "Tuuli", "Effa", "Leon", "Olivia", "Angelica", "Luxion", "Catarina", "Keith"]

def get_data_tuples():
    data_tuples = []
    
    narou_file = "data/NaroU/test/転生したらスライムだった件.json"
    character_name = "リムル＝テンペスト"
    en_name = "rimuru"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/転生したらスライムだった件.json"
    character_name = "ヴェルドラ＝テンペスト"
    en_name = "veldora"
    data_tuples.append((narou_file, character_name, en_name))
    
    narou_file = "data/NaroU/test/無職転生 - 異世界行ったら本気だす -.json"
    character_name = "ルーデウス・グレイラット"
    en_name = "rudeus"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/無職転生 - 異世界行ったら本気だす -.json"
    character_name = "ロキシー"
    en_name = "roxy"
    data_tuples.append((narou_file, character_name, en_name))
    
    narou_file = "data/NaroU/test/無職転生 - 異世界行ったら本気だす -.json"
    character_name = "シルフィエット"
    en_name = "sylphy"
    data_tuples.append((narou_file, character_name, en_name))
    
    narou_file = "data/NaroU/test/無職転生 - 異世界行ったら本気だす -.json"
    character_name = "パウロ・グレイラット"
    en_name = "paul"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/無職転生 - 異世界行ったら本気だす -.json"
    character_name = "ゼニス・グレイラット"
    en_name = "zenith"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/Ｒｅ：ゼロから始める異世界生活.json"
    character_name = "ナツキ・スバル(菜月昴)"
    en_name = "subaru"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/本好きの下剋上　～司書になるためには手段を選んでいられません～.json"
    character_name = "マイン"
    en_name = "myne"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/本好きの下剋上　～司書になるためには手段を選んでいられません～.json"
    character_name = "トゥーリ"
    en_name = "tuuli"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/本好きの下剋上　～司書になるためには手段を選んでいられません～.json"
    character_name = "マインの母"
    en_name = "effa"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/乙女ゲー世界はモブに厳しい世界です.json"
    character_name = "リオン・フォウ・バルトファルト"
    en_name = "leon"
    data_tuples.append((narou_file, character_name, en_name))
    
    narou_file = "data/NaroU/test/乙女ゲー世界はモブに厳しい世界です.json"
    character_name = "オリヴィア"
    en_name = "olivia"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/乙女ゲー世界はモブに厳しい世界です.json"
    character_name = "アンジェリカ・ラファ・レッドグレイブ"
    en_name = "angelica"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/乙女ゲー世界はモブに厳しい世界です.json"
    character_name = "ルクシオン"
    en_name = "luxion"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/乙女ゲームの破滅フラグしかない悪役令嬢に転生してしまった….json"
    character_name = "カタリナ・クラエス"
    en_name = "catarina"
    data_tuples.append((narou_file, character_name, en_name))

    narou_file = "data/NaroU/test/乙女ゲームの破滅フラグしかない悪役令嬢に転生してしまった….json"
    character_name = "キース・クラエス"
    en_name = "keith"
    data_tuples.append((narou_file, character_name, en_name))


    return data_tuples


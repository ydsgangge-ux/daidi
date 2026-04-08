"""
Layer 1 — 实业数据监控
食品板块 + 大宗商品 + 宏观货币 + 基建需求 + 全国用电量
异动识别：同比偏离度（Z-score）方法，消除季节性
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


@dataclass
class Indicator:
    name: str
    value: float
    unit: str
    yoy: Optional[float]        # 同比
    zscore: Optional[float]     # 季节性偏离度（标准差数）
    signal: str                 # UP / DOWN / NEUTRAL / ALERT
    note: str
    source: str


@dataclass
class Layer1Result:
    food: list        = field(default_factory=list)
    bulk: list        = field(default_factory=list)
    macro: list       = field(default_factory=list)
    infra: list       = field(default_factory=list)
    electricity: list = field(default_factory=list)
    logistics: list   = field(default_factory=list)
    alerts: list      = field(default_factory=list)
    score: int        = 50
    timestamp: str    = ""


def zscore_signal(zscore: Optional[float], invert: bool = False) -> str:
    """
    根据Z-score判断异动等级
    invert=True 时，高值是坏信号（如库存）
    """
    if zscore is None:
        return "NEUTRAL"
    z = -zscore if invert else zscore
    if z > 2.0:   return "ALERT_UP"
    if z > 1.5:   return "UP"
    if z < -2.0:  return "ALERT_DOWN"
    if z < -1.5:  return "DOWN"
    return "NEUTRAL"


def _safe_pct(latest, prev):
    """安全计算变动百分比"""
    if prev == 0 or pd.isna(prev) or pd.isna(latest):
        return None
    return round((latest - prev) / abs(prev) * 100, 1)


# ── 深层含义分析模板（按方向给出不同解读） ───────────────────────────────────

_ANALYSIS = {
    # ─── 食品 ───
    "农产品批发价格200指数": {
        "UP": ("农产品全面涨价，说明食品通胀抬头。可能原因：极端天气减产、"
               "化肥/饲料成本传导、猪周期上行共振。"
               "关注：生猪养殖链（牧原、温氏受益于猪价上涨）、"
               "化肥企业（云天化、盐湖股份）、种子公司（隆平高科）。"
               "风险：如果涨幅过大（>5%），央行可能收紧货币压制通胀预期"),
        "DOWN": ("农产品价格下跌，通常说明供给充裕或需求疲软。"
                 "对于养殖链是利好（饲料成本下降→利润改善），"
                 "但可能反映消费降级、居民购买力下降。"
                 "风险：如果持续下跌超过半年，说明通缩压力加大，"
                 "政策可能出台刺激消费措施"),
        "NEUTRAL": ("农产品价格平稳运行，通胀温和。对A股影响中性偏正面，"
                    "说明物价环境稳定，央行有宽松空间"),
    },
    "生猪现货价格": {
        "UP": ("猪价上涨通常意味着猪周期进入上行阶段。核心逻辑："
               "能繁母猪存栏去化→生猪供应减少→价格弹性释放。"
               "猪价一旦启动，通常持续12-18个月，涨幅可达100%+。"
               "关注：牧原股份、温氏股份（养殖龙头，弹性最大），"
               "海大集团（饲料龙头，量价齐升），"
               "生物股份（猪疫苗需求随出栏量增加）。"
               "同时关注玉米/豆粕价格——如果饲料成本也涨，养殖利润可能被压缩"),
        "DOWN": ("猪价下跌意味着养殖行业亏损扩大、产能加速去化。"
                 "短期利空养殖股，但这是下一轮猪周期的前兆。"
                 "能繁母猪存栏一旦降至低位（<4100万头），再过6-10个月猪价必然反弹。"
                 "关注：低成本产能（牧原）抗跌能力强，"
                 "饲料企业（海大）在养殖亏损期反而受益（养殖户转向自配饲料）"),
        "NEUTRAL": ("猪价在成本线附近震荡，行业微利/微亏。"
                    "等待能繁母猪存栏数据确认产能去化进度，"
                    "这是判断下一轮猪周期启动时点的核心领先指标"),
    },
    "玉米现货价格": {
        "UP": ("玉米涨价直接推高养殖成本。猪/鸡饲料中玉米占比60%+，"
               "玉米持续上涨→养殖利润被压缩→养殖户补栏积极性下降→"
               "6个月后猪价可能因供应减少而上涨。"
               "关注：种业（登海种业、隆平高科受益于粮价上涨），"
               "利空养殖（成本上升）。"
               "如果玉米涨是因为主产区干旱减产，影响可能持续整个作物年度"),
        "DOWN": ("玉米降价对养殖链是纯利好——饲料成本下降→利润率改善。"
                 "关注：牧原、温氏、圣农发展（养殖利润弹性增大）。"
                 "如果玉米持续跌破种植成本线（约2500元/吨），"
                 "农户可能减少种植→下一年度供给减少→价格反弹"),
        "NEUTRAL": ("玉米价格平稳，养殖成本端稳定。需关注后续临储拍卖政策"),
    },
    "大豆现货价格": {
        "UP": ("大豆涨价传导至豆粕（饲料蛋白核心原料）→养殖成本上升。"
               "大豆定价看巴西/美国大豆到港成本和关税政策。"
               "如果是因为中美贸易摩擦或南美减产，涨幅可能持续。"
               "关注：油脂企业（金龙鱼，但成本端承压），"
               "利空养殖（饲料成本上升）"),
        "DOWN": ("大豆降价降低豆粕/豆油成本，油脂企业利润改善。"
                 "关注：金龙鱼（原料成本下降），"
                 "利好养殖链（饲料成本下降）"),
        "NEUTRAL": ("大豆价格稳定，饲料成本端无额外压力"),
    },
    "中高端白酒景气（茅台月线）": {
        "UP": ("茅台股价上涨反映高端消费景气上行。深层含义："
               "商务宴请需求回暖（经济活跃度提升）+ 居民资产配置偏好提升。"
               "茅台是消费板块的风向标，茅台涨→"
               "五粮液、泸州老窖、山西汾酒等次高端白酒可能跟涨。"
               "但注意：如果茅台涨但社零不涨，说明只是资金抱团，"
               "不是真实消费改善，持续性存疑"),
        "DOWN": ("茅台下跌反映高端消费疲软或资金撤离消费板块。"
                 "深层原因可能是：商务活动减少（经济承压）、"
                 "消费降级（居民收入预期下降）、或机构调仓（从消费切成长）。"
                 "如果茅台连续3个月下跌，整个白酒板块需要回避。"
                 "关注：是否与PMI下行共振——如果共振，说明是真实经济问题"),
        "NEUTRAL": ("白酒景气度中性，等待节假日消费数据验证（中秋/春节是关键节点）"),
    },
    "社零总额": {
        "UP": ("社零增速超预期说明居民消费意愿回暖。深层含义："
               "就业市场改善→收入预期好转→敢于花钱→"
               "消费链全面受益（食品饮料、服装、家电、旅游）。"
               "关注：可选消费弹性最大（家电→海尔/美的，"
               "免税→中国中免，珠宝→周大生）。"
               "社零持续>5%通常对应消费牛市"),
        "DOWN": ("社零增速下滑说明消费萎缩。深层原因："
                 "居民杠杆率过高（房贷还款挤压消费）、"
                 "就业/收入预期转差、或储蓄意愿增强（预防性储蓄）。"
                 "消费链整体承压，但必选消费（酱油→海天味业，"
                 "面包→桃李面包）相对抗跌。"
                 "政策面可能出台消费券/汽车下乡等刺激措施"),
        "NEUTRAL": ("消费增速温和，没有明显方向。需关注后续就业数据和居民可支配收入"),
    },
    "女性消费景气（化妆品/医美）": {
        "UP": ("珀莱雅/爱美客月线上涨说明女性消费升级趋势强化。"
               "医美/化妆品是消费升级最敏感的品类，"
               "上涨通常意味着年轻群体可支配收入充裕+消费信心强。"
               "关注：医美（爱美客、华熙生物、华东医药），"
               "国货化妆品（珀莱雅、贝泰妮），"
               "黄金珠宝（周大生、老凤祥——婚庆需求回暖的代理）"),
        "DOWN": ("医美/化妆品下跌可能反映消费降级。"
                 "女性消费是消费信心最敏感的先行指标——"
                 "如果女生开始缩减化妆品/医美开支，说明整体消费预期在走弱。"
                 "但如果是估值回归（前期涨幅过大），可能是阶段性调整而非趋势反转"),
        "NEUTRAL": ("女性消费平稳运行，等待双十一/618等电商大促数据验证"),
    },
    # ─── 大宗 ───
    "沪铜主力合约": {
        "UP": ("铜被称为'铜博士'，是全球经济最精准的温度计。"
               "铜价上涨说明：①全球经济复苏预期增强（制造业订单增加拉动铜需求）"
               "②或供给端收缩（铜矿罢工/南美产量下降/品位下降）。"
               "铜价持续上涨→利好铜矿企业（紫金矿业、洛阳钼业，利润弹性巨大）"
               "→利好铜加工（海亮股份）→利好新能源车链（铜是电机核心材料）。"
               "风险：如果铜价上涨但LME库存也涨，说明是投机驱动而非真实需求"),
        "DOWN": ("铜价下跌是最可靠的全球衰退信号之一。"
                 "铜需求=制造业需求=全球经济活跃度。铜跌→制造业萎缩→"
                 "出口链承压、工业品全面回调。"
                 "关注：如果铜价跌破68000元/吨关键支撑，"
                 "可能引发有色板块全面抛售。但下游加工企业成本受益（电缆、空调）。"
                 "需与L2美元指数交叉验证——美元走强通常是铜价下跌的直接推手"),
        "NEUTRAL": ("铜价横盘，等待方向选择。关注LME库存变化和智利/秘鲁铜矿供应数据"),
    },
    "螺纹钢主力合约": {
        "UP": ("螺纹钢上涨意味着基建/地产施工需求回暖。"
               "深层逻辑：发改委批复项目→地方专项债发行→"
               "工地开工→螺纹钢采购放量。"
               "螺纹钢涨→利好钢企（宝钢股份、华菱钢铁），"
               "利好水泥（海螺水泥）、工程机械（三一重工）。"
               "但需区分'需求驱动涨'还是'成本驱动涨'——"
               "如果铁矿石也涨，钢企利润反而被压缩（成本挤压）"),
        "DOWN": ("螺纹钢下跌反映基建/地产需求疲软。"
                 "深层原因：地方政府财政压力大（专项债发行慢）、"
                 "房企资金链紧张（工地停工）→钢厂减产去库存。"
                 "对钢铁股利空，但可能倒逼政策加码（提前下达专项债额度）。"
                 "关注：如果螺纹钢持续阴跌+库存累积，说明需求塌方，需大幅减仓"),
        "NEUTRAL": ("螺纹钢震荡，关注水泥发运率和挖掘机开工小时数等高频数据验证需求"),
    },
    "国内汽油调价（油价代理）": {
        "UP": ("油价上涨的深层原因通常是：OPEC+减产/地缘冲突（中东战争）/"
               "全球需求超预期。油价是'通胀之母'，"
               "油价涨→运输成本涨→所有商品成本上升→通胀压力加大。"
               "关注：中国石油/中国石化（直接受益），"
               "煤化工（宝丰能源，油价涨→煤化工相对成本优势扩大），"
               "新能源车链（油价涨→电车性价比凸显，利好宁德时代、比亚迪）。"
               "风险：油价持续>80美元可能引发央行收紧货币政策"),
        "DOWN": ("油价下跌通常意味着全球需求疲软或OPEC+增产。"
                 "油价跌→运输/化工成本下降→中下游利润改善。"
                 "关注：航空（中国国航、春秋航空，燃油成本是最大成本项），"
                 "化工（万华化学），物流（顺丰控股）。"
                 "但油价暴跌（如跌破60美元）可能是全球经济衰退的信号"),
        "NEUTRAL": ("油价平稳运行，通胀压力可控。关注OPEC+会议和地缘局势变化"),
    },
    "沪金主力合约": {
        "UP": ("黄金上涨的根本驱动力有三个：①实际利率下行（美债收益率-通胀预期）"
               "②美元走弱（黄金以美元计价）③避险需求（战争/金融危机/央行去美元化）。"
               "如果三者共振，金价可能突破历史新高。"
               "关注：黄金股（紫金矿业、山东黄金、中金黄金，弹性通常是金价2-3倍），"
               "白银股（盛达资源，银的工业属性更强，弹性更大）。"
               "注意：黄金持续大涨意味着市场在定价极端风险，整体股市可能承压"),
        "DOWN": ("黄金下跌说明风险偏好回升或实际利率上行。"
                 "通常是股市的正面信号——资金从避险资产回流风险资产。"
                 "但需区分'风险偏好回升导致的下跌'（好事）和"
                 "'流动性收紧导致的下跌'（坏事，股市也会跌）。"
                 "关注：如果美债收益率上行+黄金下跌，说明流动性收紧，需降低仓位"),
        "NEUTRAL": ("金价平稳，等待美联储议息会议和通胀数据指引方向"),
    },
    "沪铝主力合约": {
        "UP": ("铝价上涨的双重驱动：①供给端：中国电解铝产能天花板（4500万吨）不可突破"
               "，供给刚性+需求增长→长期缺口。②需求端：新能源车轻量化（单车用铝量翻倍）"
               "+光伏边框（每GW铝用量约2万吨）→新能源需求持续拉动。"
               "关注：中国铝业、云铝股份、神火股份（电解铝龙头），"
               "南山铝业（航空铝材深加工）。"
               "铝的供给天花板逻辑比铜更确定，是长期结构性机会"),
        "DOWN": ("铝价下跌通常反映房地产需求塌方（建筑铝型材占铝消费30%+）。"
                 "如果仅因地产拖累下跌，而新能源需求仍在增长，"
                 "可能是阶段性错杀——中长期依然看好。"
                 "关注：中国铝业的成本优势（自备电厂+铝土矿资源），"
                 "跌到成本线附近是买入机会"),
        "NEUTRAL": ("铝价震荡，关注电解铝产能利用率和下游开工率数据"),
    },
    # ─── 宏观 ───
    "M1-M2剪刀差": {
        "UP": ("M1-M2剪刀差收窄或转正是股市最强的领先信号之一！"
               "深层逻辑：M1=企业活期存款（随时能花的钱），M2=全部存款。"
               "剪刀差收窄→企业不再囤钱→开始投资/采购/招人→经济活力恢复。"
               "历史上剪刀差转正后6-9个月，A股大概率走牛（2006、2009、2014、2020）。"
               "关注：券商（东方财富、中信证券——牛市最直接受益者），"
               "科技成长（经济复苏时弹性最大）"),
        "DOWN": ("M1-M2剪刀差扩大说明企业资金活化度下降——"
                 "钱都躺在定期存款里，不投资、不采购、不扩产。"
                 "这是经济下行/通缩的典型信号。"
                 "剪刀差<-5%通常意味着企业信心严重不足，"
                 "股市大概率偏弱。政策面可能降准降息刺激，但见效需要3-6个月。"
                 "防御策略：增配高股息（银行、电力）+ 黄金"),
        "NEUTRAL": ("剪刀差在正常区间，企业资金活化度尚可，等待更多宏观数据验证"),
    },
    "M2同比增速": {
        "UP": ("M2增速加快说明央行在放水。钱多了→部分会流入股市（资产荒逻辑）。"
               "M2>10%通常对应流动性宽松环境，利好成长股（科技、新能源、医药）。"
               "关注：如果M2持续>12%，可能出现结构性牛市"),
        "DOWN": ("M2增速放缓意味着货币收紧或信用收缩。"
                 "对股市估值形成压制，尤其是高估值板块（科技、医药）承压。"
                 "关注：如果M2<8%，需警惕系统性流动性风险"),
        "NEUTRAL": ("M2增速平稳，流动性环境中性"),
    },
    "社会融资规模增速": {
        "UP": ("社融是全系统最重要的宏观先行指标，没有之一！"
               "社融=银行贷款+信托+债券+股票融资=全社会新增借钱总量。"
               "社融加速意味着：①企业敢借钱扩产→6个月后利润增长"
               "②居民敢借钱买房/消费→地产/消费链受益"
               "③政府加大基建投资→建材/工程机械链受益。"
               "社融拐点领先GDP拐点约2个季度，领先A股底约3-6个月。"
               "关注：社融连续2个月回升→牛市起点信号，"
               "重点配置券商+基建+消费"),
        "DOWN": ("社融减速是最可靠的紧缩信号。"
                 "钱不借了→投资减少→利润下滑→股市下跌。"
                 "如果社融连续3个月下滑，需要大幅降低仓位。"
                 "防御：高股息（长江电力、中国神华）+ 国债"),
        "NEUTRAL": ("社融增速平稳，信用环境中性。关注后续专项债发行进度和政策方向"),
    },
    "制造业PMI": {
        "UP": ("PMI>50意味着制造业扩张。但关键看新订单子项——"
               "如果新订单>51+生产>51，说明量价齐升，是强复苏信号。"
               "PMI环比上升1个百分点以上=显著改善。"
               "关注：如果PMI连续3个月回升，"
               "工业自动化链（汇川技术、埃斯顿）和半导体链（中芯国际设备采购增加）"
               "通常率先受益"),
        "DOWN": ("PMI<50=制造业收缩，这是最及时的月频数据。"
                 "如果新订单<49+出厂价格<49，说明量价齐跌，需高度警惕。"
                 "PMI连续<49超过3个月=经济衰退确认。"
                 "防御策略：规避可选消费和周期股，"
                 "增配必选消费（食品）+ 公用事业（电力、水务）"),
        "NEUTRAL": ("PMI在50附近震荡，经济方向不明。关注后续PMI新订单子项变化"),
    },
    "PMI-新订单": {
        "UP": ("新订单是PMI最有预测力的子项，领先生产1-2个月。"
               "新订单回升→工厂将在下个月扩产→采购原材料→"
               "上游大宗商品价格上涨→工业品链全面受益。"
               "如果新订单>52，说明需求强劲复苏，"
               "关注：工业母机（创世纪、海天精工）、"
               "工控（汇川技术、信捷电气）"),
        "DOWN": ("新订单下降是制造业最先恶化的指标。"
                 "工厂没有新订单→开始减产→裁员→需求进一步萎缩（负循环）。"
                 "如果新订单<48连续2个月，经济下行压力极大"),
        "NEUTRAL": ("新订单平稳，等待下月数据确认趋势"),
    },
    "PMI-生产": {
        "UP": ("生产指数回升说明工厂在加班——需求好转的确认信号。"
               "如果生产>新订单，说明工厂在主动补库存，"
               "经济可能加速上行。关注：工业用电量应该同步上升作为验证"),
        "DOWN": ("生产下降说明工厂在减产。如果是被动减产（新订单在降），"
                 "问题严重；如果是主动减产去库存，可能是短期调整。"
                 "需与PMI-产成品库存交叉判断"),
        "NEUTRAL": ("生产活动平稳运行"),
    },
    "PMI-出厂价格": {
        "UP": ("出厂价格回升→企业利润率改善→2-3个月后财报利润增速上升。"
               "这是A股盈利周期最可靠的先行指标！"
               "如果出厂价格>51+原材料购进价格<50，"
               "说明'成本降+售价涨'=利润弹性最大化。"
               "关注：周期股（钢铁、化工、有色）通常最先受益于价格回暖"),
        "DOWN": ("出厂价格下降→企业利润被压缩→降价去库存→恶性循环。"
                 "如果出厂价<49+原材料购进价>51，"
                 "说明'成本涨+售价跌'=利润被两头挤压，最糟糕的组合。"
                 "需大幅规避中游制造业（毛利率会被压缩到极限）"),
        "NEUTRAL": ("出厂价格平稳，企业利润端无额外压力"),
    },
    # ─── 基建 ───
    "建筑业景气指数": {
        "UP": ("建筑业景气指数回升是基建投资加速的先行信号。"
               "发改委批复项目→地方专项债到位→工地开工→"
               "水泥/钢材/工程机械需求集中释放。"
               "关注：水泥（海螺水泥、华新水泥），"
               "工程机械（三一重工、徐工机械——挖机销量是基建最好的日频验证），"
               "管材（青龙管业、韩建河山——水利基建）。"
               "如果景气>55，说明基建力度超预期"),
        "DOWN": ("建筑业景气下降说明项目推进缓慢或资金不到位。"
                 "深层原因通常是：地方政府债务压力（没钱修），"
                 "或项目审批收紧（政策转向）。"
                 "对建材/工程机械链利空。"
                 "关注：如果连续下降，可能触发政策加码（提前下达专项债）"),
        "NEUTRAL": ("建筑业景气中性，等待地方专项债发行和重大项目开工数据"),
    },
    "房地产开发投资增速": {
        "UP": ("地产投资回升是经济全面复苏的必要条件。"
               "地产产业链占GDP约25%，上下游关联50+个行业。"
               "地产回暖→建材（水泥/玻璃/防水）、家电（空调/冰箱）、"
               "家居（定制家具/厨卫）、物业全线受益。"
               "关注：万科A、保利发展（龙头房企），"
               "东方雨虹（防水材料龙头，弹性最大），"
               "三棵树（涂料）。"
               "但需区分'新开工回升'（真复苏）和'竣工回升'（保交楼驱动，不可持续）"),
        "DOWN": ("地产投资下滑是当前中国经济最大的结构性拖累。"
                 "地产不企稳→地方政府卖地收入减少→基建没钱→"
                 "建材需求萎缩→银行坏账上升（负循环）。"
                 "如果投资增速<-10%，整个地产链需要回避。"
                 "关注：政策端是否出台'大招'（全面放开限购/国家队收储等）"
                 "——政策底通常领先市场底6-12个月"),
        "NEUTRAL": ("地产投资企稳但未明显好转，等待政策效果验证和销售端数据"),
    },
    # ─── 用电 ───
    "全国总用电量": {
        "UP": ("用电量是经济最诚实的高频指标——不可造假、不可囤积、即时反映。"
               "用电量↑说明工厂在加班、商场在营业、居民在消费。"
               "如果用电量同比>6%，经济大概率在加速增长。"
               "关注：发电企业（长江电力、华能国际），"
               "电网设备（国电南瑞、许继电气——用电量涨→电网投资扩容）。"
               "如果Z-score>2（季节性偏离超2个标准差），说明经济显著超预期"),
        "DOWN": ("用电量下降是经济减速最可靠的确认信号。"
                 "如果同比<-2%，说明工业活动明显萎缩。"
                 "需区分'基数效应'（去年同期太高）和'真实下滑'（排除基数后仍降）。"
                 "关注：如果二产用电（工业）下降但三产（服务业）上升，"
                 "说明经济在转型而非衰退"),
        "NEUTRAL": ("用电量正常波动，经济运行平稳"),
    },
    # ─── 物流 ───
    "BDI波罗的海干散货指数": {
        "UP": ("BDI上涨意味着全球大宗商品贸易活跃——"
               "铁矿石/煤炭/粮食都在大量运输。"
               "BDI涨→全球需求回暖→中国出口受益→出口制造链利好。"
               "关注：航运股（中远海控、中远海能），"
               "造船（中国船舶、中国重工——BDI涨→船东盈利→新船订单增加）。"
               "BDI>3000通常是全球经济过热的信号"),
        "DOWN": ("BDI暴跌说明全球贸易萎缩。"
                 "深层原因：中国经济放缓（铁矿石进口减少）+ 全球需求收缩。"
                 "BDI持续<1000是极端悲观信号，对应全球衰退。"
                 "关注：航运股承压，但租船费下降利好进口商成本端"),
        "NEUTRAL": ("BDI平稳，全球贸易处于正常水平"),
    },
    "BCI好望角型船运价指数": {
        "UP": ("BCI专门追踪铁矿石/煤炭运输（好望角型船只能运这些大宗散货）。"
               "BCI上涨→铁矿石海运需求旺盛→中国钢厂在大量进口铁矿石→"
               "说明钢厂看好后续钢材需求（在补库存）。"
               "利好：铁矿石企业、航运股。"
               "如果BCI涨但BDI不涨，说明是铁矿石单独拉动，"
               "需验证中国钢厂开工率是否真的在提升"),
        "DOWN": ("BCI下跌说明铁矿石运输需求萎缩。"
                 "通常意味着中国钢厂在减产（不进口了），"
                 "反映国内钢材需求疲软。"
                 "关注：如果BCI持续下跌+港口铁矿石库存累积，"
                 "说明铁矿石供过于求，钢厂利润端可能改善（原料降价）"),
        "NEUTRAL": ("BCI平稳，铁矿石海运需求正常"),
    },
}


def _analysis_note(name: str, signal: str, default_note: str = "") -> str:
    """根据指标名称和信号方向，返回详细的深层含义分析"""
    templates = _ANALYSIS.get(name, {})
    direction = signal.upper()
    if "ALERT" in direction:
        direction = "UP" if "UP" in direction else "DOWN"
    return templates.get(direction, default_note)


# ── 食品板块 ──────────────────────────────────────────────────────────────────

def get_food_data() -> list:
    indicators = []

    # 1. 生鲜乳收购价（搜猪网/农产品价格指数）
    try:
        df = ak.macro_china_agricultural_product()
        if df is not None and len(df) > 0:
            df.columns = [c.strip() for c in df.columns]
            # 返回的是农产品批发价格200指数（综合）
            df["最新值"] = pd.to_numeric(df["最新值"], errors="coerce")
            series = df["最新值"].dropna()
            latest = float(series.iloc[-1])
            prev   = float(series.iloc[-26]) if len(series) >= 26 else float(series.iloc[0])
            yoy    = (latest - prev) / prev * 100
            indicators.append(Indicator(
                name="农产品批发价格200指数", value=round(latest, 2), unit="点",
                yoy=round(yoy, 1), zscore=None,
                signal="UP" if yoy > 3 else ("DOWN" if yoy < -3 else "NEUTRAL"),
                note=_analysis_note("农产品批发价格200指数",
                      "UP" if yoy > 3 else ("DOWN" if yoy < -3 else "NEUTRAL")),
                source="农业农村部"
            ))
    except Exception:
        pass

    # 2. 生猪价格（搜猪网现货指数）
    try:
        df = ak.index_hog_spot_price()
        if df is not None and len(df) > 5:
            df.columns = [c.strip() for c in df.columns]
            df["指数"] = pd.to_numeric(df["指数"], errors="coerce")
            series = df["指数"].dropna()
            latest = float(series.iloc[-1])
            # 12个月均线作为基准
            df["12个月均线"] = pd.to_numeric(df["12个月均线"], errors="coerce")
            ma12 = df["12个月均线"].dropna()
            if len(ma12) > 0:
                benchmark = float(ma12.iloc[-1])
                yoy = round((latest - benchmark) / benchmark * 100, 1)
            else:
                yoy = None
            # 成交均价
            avg_price = None
            if "成交均价" in df.columns:
                df["成交均价"] = pd.to_numeric(df["成交均价"], errors="coerce")
                p = df["成交均价"].dropna()
                if len(p) > 0:
                    avg_price = float(p.iloc[-1])
            display_val = avg_price if avg_price and avg_price > 0 else latest
            display_unit = "元/kg" if avg_price else "点"
            indicators.append(Indicator(
                name="生猪现货价格", value=round(display_val, 2), unit=display_unit,
                yoy=yoy, zscore=None,
                signal="UP" if (yoy and yoy > 10) else ("DOWN" if (yoy and yoy < -10) else "NEUTRAL"),
                note=_analysis_note("生猪现货价格",
                      "UP" if (yoy and yoy > 10) else ("DOWN" if (yoy and yoy < -10) else "NEUTRAL")),
                source="搜猪网"
            ))
    except Exception:
        pass

    # 3. 玉米现货价格
    try:
        df = ak.spot_corn_price_soozhu()
        if df is not None and len(df) > 0:
            df.columns = [c.strip() for c in df.columns]
            df["价格"] = pd.to_numeric(df["价格"], errors="coerce")
            latest = float(df["价格"].dropna().iloc[-1])
            indicators.append(Indicator(
                name="玉米现货价格", value=round(latest, 2), unit="元/kg",
                yoy=None, zscore=None, signal="NEUTRAL",
                note=_analysis_note("玉米现货价格", "NEUTRAL"),
                source="搜猪网"
            ))
    except Exception:
        pass

    # 4. 大豆现货价格
    try:
        df = ak.spot_soybean_price_soozhu()
        if df is not None and len(df) > 0:
            df.columns = [c.strip() for c in df.columns]
            df["价格"] = pd.to_numeric(df["价格"], errors="coerce")
            latest = float(df["价格"].dropna().iloc[-1])
            indicators.append(Indicator(
                name="大豆现货价格", value=round(latest, 2), unit="元/kg",
                yoy=None, zscore=None, signal="NEUTRAL",
                note=_analysis_note("大豆现货价格", "NEUTRAL"),
                source="搜猪网"
            ))
    except Exception:
        pass

    # 5. 白酒：用贵州茅台月线作为行业景气代理
    try:
        df = ak.stock_zh_a_hist(symbol="600519", period="monthly",
                                start_date="20230101", adjust="qfq")
        if df is not None and len(df) > 2:
            df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
            latest = float(df["收盘"].iloc[-1])
            prev   = float(df["收盘"].iloc[-4]) if len(df) >= 4 else float(df["收盘"].iloc[0])
            chg    = (latest - prev) / prev * 100
            indicators.append(Indicator(
                name="中高端白酒景气（茅台月线）",
                value=round(latest, 2), unit="元",
                yoy=round(chg, 1), zscore=None,
                signal="UP" if chg > 5 else ("DOWN" if chg < -5 else "NEUTRAL"),
                note=_analysis_note("中高端白酒景气（茅台月线）",
                      "UP" if chg > 5 else ("DOWN" if chg < -5 else "NEUTRAL")),
                source="A股行情"
            ))
    except Exception:
        pass

    # 6. 社会消费品零售总额（消费整体景气）
    try:
        df = ak.macro_china_consumer_goods_retail()
        if df is not None and len(df) > 2:
            df["当月"] = pd.to_numeric(df["当月"], errors="coerce")
            df["同比增长"] = pd.to_numeric(df["同比增长"], errors="coerce")
            df = df.dropna(subset=["当月", "同比增长"])
            latest = float(df["当月"].iloc[-1])
            yoy = float(df["同比增长"].iloc[-1])
            indicators.append(Indicator(
                name="社零总额", value=round(latest, 1), unit="亿元",
                yoy=yoy, zscore=None,
                signal="UP" if yoy > 5 else ("DOWN" if yoy < 2 else "NEUTRAL"),
                note=_analysis_note("社零总额",
                      "UP" if yoy > 5 else ("DOWN" if yoy < 2 else "NEUTRAL")),
                source="国家统计局"
            ))
    except Exception:
        pass

    # 7. 女性消费景气（用珀莱雅/爱美客月线做代理）
    # 珀莱雅 = 国货化妆品龙头，爱美客 = 医美龙头，收入提升最先反映在这些品类
    female_consumption = []
    for code, label in [("603605", "珀莱雅"), ("300896", "爱美客")]:
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="monthly",
                                    start_date="20230101", adjust="qfq")
            if df is not None and len(df) > 2:
                df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
                series = df["收盘"].dropna()
                latest = float(series.iloc[-1])
                prev12 = float(series.iloc[-13]) if len(series) >= 13 else float(series.iloc[0])
                chg = round((latest - prev12) / prev12 * 100, 1)
                female_consumption.append(f"{label}{chg:+.1f}%")
        except Exception:
            pass
    if female_consumption:
        female_signal = (
            "UP" if any("+" in x and float(x.split("+")[1].replace("%","")) > 10 for x in female_consumption)
            else "DOWN" if any("-" in x and float(x.split("-")[1].replace("%","")) < -15 for x in female_consumption)
            else "NEUTRAL"
        )
        indicators.append(Indicator(
            name="女性消费景气（化妆品/医美）",
            value=f"珀莱雅+爱美客", unit="月线同比",
            yoy=None, zscore=None,
            signal=female_signal,
            note=_analysis_note("女性消费景气（化妆品/医美）", female_signal),
            source="A股行情"
        ))

    return indicators


# ── 大宗商品 ─────────────────────────────────────────────────────────────────

def get_bulk_data() -> list:
    indicators = []

    # 1. 铜价（上期所）
    try:
        df = ak.futures_zh_realtime(symbol="铜")
        if df is not None and len(df) > 0:
            price = pd.to_numeric(df["最新价"].iloc[0], errors="coerce")
            chg   = pd.to_numeric(df["涨跌幅"].iloc[0], errors="coerce")
            if pd.notna(price):
                copper_signal = "UP" if float(chg) > 1 else ("DOWN" if float(chg) < -1 else "NEUTRAL")
                indicators.append(Indicator(
                    yoy=None, zscore=None,
                    signal=copper_signal,
                    note=_analysis_note("沪铜主力合约", copper_signal),
                    source="上期所"
                ))
    except Exception:
        pass

    # 2. 螺纹钢
    try:
        df = ak.futures_zh_realtime(symbol="螺纹钢")
        if df is not None and len(df) > 0:
            price = pd.to_numeric(df["最新价"].iloc[0], errors="coerce")
            chg   = pd.to_numeric(df["涨跌幅"].iloc[0], errors="coerce")
            if pd.notna(price):
                rebar_signal = "UP" if float(chg) > 1 else ("DOWN" if float(chg) < -1 else "NEUTRAL")
                indicators.append(Indicator(
                    name="螺纹钢主力合约", value=round(float(price), 0), unit="元/吨",
                    yoy=None, zscore=None,
                    signal=rebar_signal,
                    note=_analysis_note("螺纹钢主力合约", rebar_signal),
                    source="上期所"
                ))
    except Exception:
        pass

    # 3. 原油（国内汽柴油调价作为油价趋势代理）
    try:
        df = ak.energy_oil_hist()
        if df is not None and len(df) > 5:
            df.columns = [c.strip() for c in df.columns]
            # 使用汽油价格变化作为油价趋势代理
            price_col = '汽油价格' if '汽油价格' in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            chg_col = '汽油涨跌' if '汽油涨跌' in df.columns else None
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
            latest = float(df[price_col].dropna().iloc[-1])
            prev   = float(df[price_col].dropna().iloc[-2]) if len(df[price_col].dropna()) >= 2 else latest
            chg    = _safe_pct(latest, prev)
            # 取最近几次调价的累计变化
            if chg_col:
                df[chg_col] = pd.to_numeric(df[chg_col], errors="coerce")
                recent_chgs = df[chg_col].dropna().tail(3)
                if len(recent_chgs) > 0:
                    total_chg = sum(recent_chgs)
                    chg = round(total_chg / abs(prev) * 100, 1) if prev != 0 else 0
            oil_signal = "UP" if (chg and chg > 3) else ("DOWN" if (chg and chg < -3) else "NEUTRAL")
            indicators.append(Indicator(
                name="国内汽油调价（油价代理）", value=round(latest, 0), unit="元/吨",
                yoy=chg, zscore=None,
                signal=oil_signal,
                note=_analysis_note("国内汽油调价（油价代理）", oil_signal),
                source="发改委"
            ))
    except Exception:
        pass

    # 4. 黄金（沪金）
    try:
        df = ak.futures_zh_realtime(symbol="黄金")
        if df is not None and len(df) > 0:
            price = pd.to_numeric(df["最新价"].iloc[0], errors="coerce")
            chg   = pd.to_numeric(df["涨跌幅"].iloc[0], errors="coerce")
            if pd.notna(price):
                gold_signal = "UP" if float(chg) > 0.5 else ("DOWN" if float(chg) < -0.5 else "NEUTRAL")
                indicators.append(Indicator(
                    name="沪金主力合约", value=round(float(price), 2), unit="元/克",
                    yoy=None, zscore=None,
                    signal=gold_signal,
                    note=_analysis_note("沪金主力合约", gold_signal),
                    source="上期所"
                ))
    except Exception:
        pass

    # 5. 铝价（上期所）——新能源车轻量化+电力成本关联
    try:
        df = ak.futures_zh_realtime(symbol="铝")
        if df is not None and len(df) > 0:
            price = pd.to_numeric(df["最新价"].iloc[0], errors="coerce")
            chg   = pd.to_numeric(df["涨跌幅"].iloc[0], errors="coerce")
            if pd.notna(price):
                alum_signal = "UP" if float(chg) > 1 else ("DOWN" if float(chg) < -1 else "NEUTRAL")
                indicators.append(Indicator(
                    name="沪铝主力合约", value=round(float(price), 0), unit="元/吨",
                    yoy=None, zscore=None,
                    signal=alum_signal,
                    note=_analysis_note("沪铝主力合约", alum_signal),
                    source="上期所"
                ))
    except Exception:
        pass

    return indicators


# ── 物流运价 ─────────────────────────────────────────────────────────────────

def get_logistics_data() -> list:
    indicators = []

    # BDI 波罗的海干散货指数
    try:
        df = ak.macro_shipping_bdi()
        if df is not None and len(df) > 2:
            df.columns = [c.strip() for c in df.columns]
            val_col = [c for c in df.columns if "最新值" in c or "收盘" in c or "BDI" in c.upper()]
            if not val_col:
                val_col = [df.columns[1]]
            df[val_col[0]] = pd.to_numeric(df[val_col[0]], errors="coerce")
            series = df[val_col[0]].dropna()
            latest = float(series.iloc[-1])
            prev   = float(series.iloc[-5]) if len(series) >= 5 else latest
            chg    = (latest - prev) / prev * 100 if prev != 0 else 0
            bdi_signal = "UP" if chg > 10 else ("DOWN" if chg < -10 else "NEUTRAL")
            indicators.append(Indicator(
                name="BDI波罗的海干散货指数", value=round(latest, 0), unit="点",
                yoy=round(chg, 1), zscore=None,
                signal=bdi_signal,
                note=_analysis_note("BDI波罗的海干散货指数", bdi_signal),
                source="波罗的海交易所"
            ))
    except Exception:
        pass

    # BCI 波罗的海好望角型船运价指数（铁矿石专用）
    try:
        df = ak.macro_shipping_bci()
        if df is not None and len(df) > 2:
            df.columns = [c.strip() for c in df.columns]
            val_col = [c for c in df.columns if "最新值" in c or "收盘" in c]
            if not val_col:
                val_col = [df.columns[1]]
            df[val_col[0]] = pd.to_numeric(df[val_col[0]], errors="coerce")
            series = df[val_col[0]].dropna()
            latest = float(series.iloc[-1])
            prev   = float(series.iloc[-5]) if len(series) >= 5 else latest
            chg    = (latest - prev) / prev * 100 if prev != 0 else 0
            bci_signal = "UP" if chg > 15 else ("DOWN" if chg < -15 else "NEUTRAL")
            indicators.append(Indicator(
                name="BCI好望角型船运价指数", value=round(latest, 0), unit="点",
                yoy=round(chg, 1), zscore=None,
                signal=bci_signal,
                note=_analysis_note("BCI好望角型船运价指数", bci_signal),
                source="波罗的海交易所"
            ))
    except Exception:
        pass

    return indicators


# ── 宏观货币 ─────────────────────────────────────────────────────────────────

def get_macro_data() -> list:
    indicators = []

    # 1. M1/M2
    try:
        df = ak.macro_china_money_supply()
        if df is not None and len(df) > 2:
            df.columns = [c.strip() for c in df.columns]
            m1_cols = [c for c in df.columns if "M1" in c and "同比" in c]
            m2_cols = [c for c in df.columns if "M2" in c and "同比" in c]
            if m1_cols and m2_cols:
                df[m1_cols[0]] = pd.to_numeric(df[m1_cols[0]], errors="coerce")
                df[m2_cols[0]] = pd.to_numeric(df[m2_cols[0]], errors="coerce")
                m1 = float(df[m1_cols[0]].dropna().iloc[-1])
                m2 = float(df[m2_cols[0]].dropna().iloc[-1])
                spread = round(m1 - m2, 2)
                ms_signal = "UP" if spread > -2 else ("DOWN" if spread < -5 else "NEUTRAL")
                m2_signal = "UP" if m2 > 9 else ("DOWN" if m2 < 7 else "NEUTRAL")
                indicators.append(Indicator(
                    name="M1-M2剪刀差", value=spread, unit="%",
                    yoy=None, zscore=None,
                    signal=ms_signal,
                    note=_analysis_note("M1-M2剪刀差", ms_signal),
                    source="中国人民银行"
                ))
                indicators.append(Indicator(
                    name="M2同比增速", value=round(m2, 2), unit="%",
                    yoy=None, zscore=None,
                    signal=m2_signal,
                    note=_analysis_note("M2同比增速", m2_signal),
                    source="中国人民银行"
                ))
    except Exception:
        pass

    # 2. 社融
    try:
        df = ak.macro_china_shrzgm()
        if df is not None and len(df) > 2:
            df.columns = [c.strip() for c in df.columns]
            yoy_cols = [c for c in df.columns if "同比" in c or "增速" in c]
            if yoy_cols:
                df[yoy_cols[0]] = pd.to_numeric(df[yoy_cols[0]], errors="coerce")
                val = float(df[yoy_cols[0]].dropna().iloc[-1])
                indicators.append(Indicator(
                    name="社会融资规模增速", value=round(val, 2), unit="%",
                    yoy=None, zscore=None,
                    signal="UP" if val > 10 else ("DOWN" if val < 8 else "NEUTRAL"),
                    note=_analysis_note("社会融资规模增速",
                          "UP" if val > 10 else ("DOWN" if val < 8 else "NEUTRAL")),
                    source="中国人民银行"
                ))
    except Exception:
        pass

    # 3. PMI制造业（总值）
    try:
        df = ak.macro_china_pmi_yearly()
        if df is not None and len(df) > 1:
            df.columns = [c.strip() for c in df.columns]
            val_col = [c for c in df.columns if "制造业" in c or "PMI" in c]
            if val_col:
                df[val_col[0]] = pd.to_numeric(df[val_col[0]], errors="coerce")
                val = float(df[val_col[0]].dropna().iloc[-1])
                indicators.append(Indicator(
                    name="制造业PMI", value=round(val, 1), unit="",
                    yoy=None, zscore=None,
                    signal="UP" if val > 51 else ("DOWN" if val < 49 else "NEUTRAL"),
                    note=_analysis_note("制造业PMI",
                          "UP" if val > 51 else ("DOWN" if val < 49 else "NEUTRAL")),
                    source="国家统计局"
                ))
    except Exception:
        pass

    # 4. PMI子项：新订单 + 生产 + 出厂价格
    try:
        df = ak.macro_china_pmi(year=2025)
        if df is not None and len(df) > 0:
            df.columns = [c.strip() for c in df.columns]
            sub_items = {
                "新订单": "新订单",
                "生产": "生产",
                "出厂价格": "出厂价格",
            }
            for label, keyword in sub_items.items():
                col = [c for c in df.columns if keyword in c]
                if col:
                    df[col[0]] = pd.to_numeric(df[col[0]], errors="coerce")
                    val = float(df[col[0]].dropna().iloc[-1])
                    sig = "UP" if val > 51 else ("DOWN" if val < 49 else "NEUTRAL")
                    indicators.append(Indicator(
                        name=f"PMI-{label}", value=round(val, 1), unit="",
                        yoy=None, zscore=None,
                        signal=sig,
                        note=_analysis_note(f"PMI-{label}", sig),
                        source="国家统计局"
                    ))
    except Exception:
        pass

    return indicators


# ── 基建需求 ─────────────────────────────────────────────────────────────────

def get_infra_data() -> list:
    indicators = []

    # 建筑业商务活动指数（中国物流与采购联合会）
    try:
        df = ak.macro_china_construction_index()
        if df is not None and len(df) > 1:
            df.columns = [c.strip() for c in df.columns]
            val_col = '最新值' if '最新值' in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            chg_col = '涨跌幅' if '涨跌幅' in df.columns else None
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
            latest = float(df[val_col].dropna().iloc[-1])
            chg = None
            if chg_col:
                df[chg_col] = pd.to_numeric(df[chg_col], errors="coerce")
                c = df[chg_col].dropna()
                if len(c) > 0:
                    chg = round(float(c.iloc[-1]), 1)
            infra_signal = "UP" if (chg and chg > 2) else ("DOWN" if (chg and chg < -2) else "NEUTRAL")
            indicators.append(Indicator(
                name="建筑业景气指数", value=round(latest, 1), unit="点",
                yoy=chg, zscore=None,
                signal=infra_signal,
                note=_analysis_note("建筑业景气指数", infra_signal),
                source="中国物流与采购联合会"
            ))
    except Exception:
        pass

    # 房地产开发投资
    try:
        df = ak.macro_china_real_estate()
        if df is not None and len(df) > 1:
            df.columns = [c.strip() for c in df.columns]
            yoy_col = [c for c in df.columns if "同比" in c]
            if yoy_col:
                df[yoy_col[0]] = pd.to_numeric(df[yoy_col[0]], errors="coerce")
                val = float(df[yoy_col[0]].dropna().iloc[-1])
                indicators.append(Indicator(
                    name="房地产开发投资增速", value=round(val, 1), unit="%",
                    yoy=None, zscore=None,
                    signal="DOWN" if val < -5 else ("UP" if val > 3 else "NEUTRAL"),
                    note=_analysis_note("房地产开发投资增速",
                          "DOWN" if val < -5 else ("UP" if val > 3 else "NEUTRAL")),
                    source="国家统计局"
                ))
    except Exception:
        pass

    return indicators


# ── 全国用电量 ────────────────────────────────────────────────────────────────

def get_electricity_data() -> list:
    indicators = []
    try:
        df = ak.macro_china_society_electricity()
        if df is not None and len(df) > 1:
            df.columns = [c.strip() for c in df.columns]

            # 总用电量
            total_col = [c for c in df.columns if "全社会用电量" == c and "同比" not in c]
            if total_col:
                df[total_col[0]] = pd.to_numeric(df[total_col[0]], errors="coerce")
                series = df[total_col[0]].dropna()
                if len(series) > 0:
                    latest_val = float(series.iloc[-1])

                    # 同比增速（已有列）
                    yoy_col = [c for c in df.columns if "全社会用电量同比" in c]
                    yoy_val = None
                    if yoy_col:
                        df[yoy_col[0]] = pd.to_numeric(df[yoy_col[0]], errors="coerce")
                        y = df[yoy_col[0]].dropna()
                        if len(y) > 0:
                            yoy_val = round(float(y.iloc[-1]), 2)

                    # Z-score 季节性偏离
                    z = 0
                    if len(series) >= 12:
                        hist_vals = [float(series.iloc[i]) for i in range(len(series)-1)]
                        if len(hist_vals) >= 2:
                            mu = np.mean(hist_vals)
                            sigma = np.std(hist_vals)
                            z = (latest_val - mu) / sigma if sigma > 0 else 0

                    signal = zscore_signal(z)
                    e_note = _analysis_note("全国总用电量", signal)
                    if abs(z) > 1.5:
                        e_note += f"（当前季节性偏离{z:+.2f}σ，超出正常区间！）"
                    else:
                        e_note += f"（季节性偏离{z:+.2f}σ，在正常范围内）"
                    indicators.append(Indicator(
                        name="全国总用电量", value=round(latest_val, 0), unit="万千瓦时",
                        yoy=yoy_val, zscore=round(z, 2), signal=signal,
                        note=e_note,
                        source="国家能源局"
                    ))
    except Exception as e:
        indicators.append(Indicator(
            name="全国总用电量", value=0, unit="万千瓦时",
            yoy=None, zscore=None, signal="NEUTRAL",
            note=f"数据获取失败: {str(e)[:50]}",
            source="国家能源局"
        ))

    # 分行业用电量（同一数据源有详细分项列）
    try:
        df = ak.macro_china_society_electricity()
        if df is not None and len(df) > 1:
            df.columns = [c.strip() for c in df.columns]
            industry_map = {
                "第一产业用电量": ("第一产业", "农业用电趋势"),
                "第二产业用电量": ("第二产业(工业)", "制造业/采矿业用电，工业景气核心指标"),
                "第三产业用电量": ("第三产业(服务业)", "服务业/商业/IT用电，消费景气代理"),
                "城乡居民生活用电量合计": ("城乡居民生活", "居民消费能力代理"),
            }
            for col, (label, note) in industry_map.items():
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        series = df[col].dropna()
                        if len(series) >= 2:
                            latest = float(series.iloc[-1])
                            # 找对应的同比列
                            yoy_col_name = col + "同比"
                            yoy = None
                            if yoy_col_name in df.columns:
                                df[yoy_col_name] = pd.to_numeric(df[yoy_col_name], errors="coerce")
                                y = df[yoy_col_name].dropna()
                                if len(y) > 0:
                                    yoy = round(float(y.iloc[-1]), 2)
                            indicators.append(Indicator(
                                name=f"用电量-{label}", value=round(latest, 0), unit="万千瓦时",
                                yoy=yoy, zscore=None,
                                signal="UP" if (yoy and yoy > 5) else ("DOWN" if (yoy and yoy < -3) else "NEUTRAL"),
                                note=note, source="国家能源局"
                            ))
                    except Exception:
                        continue
    except Exception:
        pass

    return indicators


# ── 汇总评分 ─────────────────────────────────────────────────────────────────

def score_layer1(result: Layer1Result) -> int:
    all_inds = (result.food + result.bulk + result.macro
                + result.infra + result.electricity + result.logistics)
    if not all_inds:
        return 50
    up   = sum(1 for i in all_inds if i.signal in ("UP", "ALERT_UP"))
    down = sum(1 for i in all_inds if i.signal in ("DOWN", "ALERT_DOWN"))
    total = len(all_inds)
    return int(50 + (up - down) / total * 50)


def run_layer1() -> Layer1Result:
    result = Layer1Result(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))

    result.food        = get_food_data()
    result.bulk        = get_bulk_data()
    result.macro       = get_macro_data()
    result.infra       = get_infra_data()
    result.electricity = get_electricity_data()
    result.logistics   = get_logistics_data()

    # 异动提醒
    all_inds = (result.food + result.bulk + result.macro
                + result.infra + result.electricity + result.logistics)
    result.alerts = [i for i in all_inds if "ALERT" in i.signal]
    result.score  = score_layer1(result)

    return result

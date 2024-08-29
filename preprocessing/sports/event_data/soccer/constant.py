from bidict import bidict

"""
共通
"""

FREQUENCY_HZ_ORIGINAL = 25
FREQUENCY_HZ_CONVERTED = 10


"""
データスタジアムのカラム
"""
# データスタジアムのカラム"攻撃方向"とトラッキングデータの座標正負との関係
ATTACKING_DIRECTION_PLUS = 1 # 1の場合、正方向に攻撃
ATTACKING_DIRECTION_MINUS = 2 # 2の場合、負方向に攻撃

# データスタジアムのカラム"ホームアウェイF"とチームの関係
F_HOME_AWAY_BALL = 0
F_HOME_AWAY_HOME = 1 
F_HOME_AWAY_AWAY = 2 

# play.csv（イベントデータ）から抽出するイベント
ACTION_NAME_FOR_ATTACK_ANALYSIS_LIST = [
    # 自チームイベント
    'シュート'
    ,'ホームパス'
    ,'アウェイパス'
    ,'スルーパス'
    ,'フィード'
    ,'クロス'
    ,'GK'
    ,'CK'
    ,'キックオフ'
    ,'スローイン'
    ,'ドリブル'
    # ,'直接FK'
    # ,'間接FK'

    # 敵チームイベント
    , 'トラップ'
    , 'クリア'
    , 'ブロック'
]


# アクション一覧
# usage
# print(ACTION_BIDICT['CK']) # 1
# print(ACTION_BIDICT.inverse[1]) # CK 

ACTION_BIDICT = bidict({
    'CK': 1
    ,'シュート': 2
    ,'キックオフ': 3
    ,'クロス': 4
    ,'ハンドクリア': 5
    ,'タッチ': 6
    ,'ボールアウト': 7
    ,'間接FK': 8
    ,'PK': 9
    ,'タックル': 10
    ,'試合中断(試合中)': 11
    ,'フリック>オン': 12
    ,'退場(レッド)': 13
    ,'オフサイド': 14
    ,'ポスト/バー': 15
    ,'ファウル受ける': 16
    ,'前半開始': 17
    ,'ホームパス': 18
    ,'トラップ': 19
    ,'クリア': 20
    ,'直接FK': 21
    ,'前半終了': 22
    ,'後半終了': 23
    ,'オウンゴール': 24
    ,'警告(イエロー)': 25
    ,'ドリブル': 26
    ,'ファウルする': 27
    ,'スルーパス': 28
    ,'キャッチ': 29
    ,'フィード': 30
    ,'アウェイパス': 31
    ,'交代': 32
    ,'後半開始': 33
    ,'インターセプト': 34
    ,'ドロップボール': 35
    ,'GK': 36
    ,'ブロック': 37
    ,'スローイン': 38
})

# アクション優先度（大きいほど高い）
# 同フレームに複数のアクションが紐付いている際、どのアクションを残すかを決定
ACTION_PRIORITY = {
    # 自チームイベント
    'シュート': 10
    ,'ホームパス': 10
    ,'アウェイパス': 10
    ,'スルーパス': 10
    ,'フィード': 10
    ,'クロス': 10
    ,'GK': 8
    ,'CK': 8
    ,'キックオフ': 8
    ,'スローイン': 8
    ,'ドリブル': 8
    # ,'直接FK'
    # ,'間接FK'

    # 敵チームイベント
    , 'トラップ': 6
    , 'クリア': 9 # シュート、クリア重複あり

    # 共通イベント
    , 'ブロック': 4 # トラップ、ブロック重複あり
}

TEAM_BIDICT = bidict({
    122: '浦和レッズ'
    , 128: 'ガンバ大阪'
    , 124: '横浜Ｆ・マリノス'
    , 127: '名古屋グランパス'
    , 126: '清水エスパルス'
    , 133: 'セレッソ大阪'
    , 136: 'ヴィッセル神戸'
    , 30528: '松本山雅ＦＣ'
    , 120: '鹿島アントラーズ'
    , 238: 'ベガルタ仙台'
    , 129: 'サンフレッチェ広島'
    , 131: 'ジュビロ磐田'
    , 207: '大分トリニータ'
    , 86: '川崎フロンタ>ーレ'
    , 86: '川崎フロンターレ'
    , 276: '北海道コンサドーレ札幌'
    , 270: 'ＦＣ東京'
    , 130: '湘南ベルマーレ'
    , 269: 'サガン鳥栖'
})


N_AGENTS = 22
EXTRA_FRAME = 4 

FIELD_LENGTH = 105.0  # unit: meters
FIELD_WIDTH = 68.0  # unit: meters
GOAL_WIDTH = 7.32  # unit: meters
PENALTY_X = 105.0/2-16.5 # left point (unit: meters)
PENALTY_Y = 40.32 # upper point (unit: meters)

# for gfootball
FIELD_LENGTH_GRF = 1*2 
FIELD_WIDTH_GRF = 0.42*2
GOAL_WIDTH_GRF = 0.044*2

STOP_THRESHOLD = 0.1 # unit: m/s
SPRINT_THRESHOLD = 24000/3600 # unit: m/s (24 km/h)
LONGPASS_THRESHOLD = 30 # unit: meters
HIGHPASS_AGENT_THRESHOLD = 1 # unit: meters
BALL_KEEP_THRESHOLD = 1 # unit: m
SEED = 42

# super mini map for gfootball
SMM_WIDTH = 96
SMM_HEIGHT = 72

SMM_LAYERS = ['left_team', 'right_team', 'ball', 'active']

# Normalized minimap coordinates
MINIMAP_NORM_X_MIN = -1.0
MINIMAP_NORM_X_MAX = 1.0
MINIMAP_NORM_Y_MIN = -1.0 / 2.25
MINIMAP_NORM_Y_MAX = 1.0 / 2.25

MARKER_VALUE = 255

# GFootbal actions
ACTION_GRF_19 = bidict({
    'idle': 0
    ,'left': 1
    ,'top_left': 2
    ,'top': 3
    ,'top_right': 4
    ,'right': 5
    ,'bottom_right': 6
    ,'bottom': 7
    ,'bottom_left': 8
    ,'long_pass': 9
    ,'high_pass': 10
    ,'short_pass': 11
    ,'shot': 12
    ,'sprint': 13
    ,'release_direction': 14
    ,'release_sprint': 15
    ,'sliding': 16
    ,'dribble': 17
    ,'release_dribble': 18 # ,'builtin_ai ': 19
})

ACTION_GRF_14 = bidict({
    'idle': 0
    ,'left': 1
    ,'top_left': 2
    ,'top': 3
    ,'top_right': 4
    ,'right': 5
    ,'bottom_right': 6
    ,'bottom': 7
    ,'bottom_left': 8
    ,'pass': 9
    ,'shot': 10
    ,'sprint': 11
    ,'release_direction': 12
    ,'release_sprint': 13
})
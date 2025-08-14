from .ultimatetrack_preprocessing.preprocessing import (
    process_tracking_data as process_ultimatetrack_tracking_data,
)


class Ultimate_tracking_data:
    """
    Ultimate Track データ処理クラス

    フィールド仕様:
    - 長さ: 94m
    - 幅: 37m
    - フレームレート: 15fps
    """

    # Ultimate Track specifications
    FIELD_LENGTH = 94.0  # meters
    FIELD_WIDTH = 37.0  # meters
    FRAME_RATE = 15  # fps
    PLAYERS_PER_TEAM = 7

    @staticmethod
    def process_ultimatetrack_tracking_data(*args, **kwargs):
        """Ultimate Track トラッキングデータの処理"""
        tracking_offense, tracking_defense, tracking_disc, team_info_df = (
            process_ultimatetrack_tracking_data(*args, **kwargs)
        )
        return tracking_offense, tracking_defense, tracking_disc, team_info_df

    @classmethod
    def get_field_info(cls):
        """フィールド情報を取得"""
        return {
            "length": cls.FIELD_LENGTH,
            "width": cls.FIELD_WIDTH,
            "frame_rate": cls.FRAME_RATE,
            "players_per_team": cls.PLAYERS_PER_TEAM,
        }

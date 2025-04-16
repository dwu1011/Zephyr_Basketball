from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BasketballCourtConfiguration:
    """
    Configuration for a basketball court.
    The court is defined by its width, length, and various markings.
    """
    # Dimensions in centimeters
    # The dimensions are based on FIBA regulations
    # https://www.fiba.basketball/documents/2021/8/3/2021_FIBA_Basketball_Court_Construction_Guide.pdf
    # and the official FIBA court dimensions


    width: int = 1524  # [cm]
    length: int = 2440  # [cm]


    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            (0, 0),  # 1
            (0, self.width),  # 2
            (self.length, self.width),  # 3
            (self.length, 0),  # 4
        ]

    # edges: List[Tuple[int, int]] = field(default_factory=lambda: [
    #     (1, 2), (2, 3), (3, 4), (4, 1)
    # ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05",
        "11", "12", "13", "14", "15",
        "21", "22", "23", "24", "25",
        "31", "32", "33", "34", "35",
        "41", "42", "43", "44", "45",
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF8000", "#FF8000", "#FF8000", "#FF8000", "#FF8000",  # center line
        "#C7FC00", "#C7FC00", "#C7FC00", "#C7FC00", "#C7FC00",  # left paint
        "#FE0056", "#FE0056", "#FE0056", "#FE0056", "#FE0056",  # left three point
        "#8622FF", "#8622FF", "#8622FF", "#8622FF", "#8622FF",  # right paint
        "#00FFCE", "#00FFCE", "#00FFCE", "#00FFCE", "#00FFCE",  # right three point
    ])


# @dataclass
# class SoccerPitchConfiguration:
#     width: int = 7000  # [cm]
#     length: int = 12000  # [cm]
#     penalty_box_width: int = 4100  # [cm]
#     penalty_box_length: int = 2015  # [cm]
#     goal_box_width: int = 1832  # [cm]
#     goal_box_length: int = 550  # [cm]
#     centre_circle_radius: int = 915  # [cm]
#     penalty_spot_distance: int = 1100  # [cm]

#     @property
#     def vertices(self) -> List[Tuple[int, int]]:
#         return [
#             (0, 0),  # 1
#             (0, (self.width - self.penalty_box_width) / 2),  # 2
#             (0, (self.width - self.goal_box_width) / 2),  # 3
#             (0, (self.width + self.goal_box_width) / 2),  # 4
#             (0, (self.width + self.penalty_box_width) / 2),  # 5
#             (0, self.width),  # 6
#             (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
#             (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
#             (self.penalty_spot_distance, self.width / 2),  # 9
#             (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
#             (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
#             (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
#             (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
#             (self.length / 2, 0),  # 14
#             (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
#             (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
#             (self.length / 2, self.width),  # 17
#             (
#                 self.length - self.penalty_box_length,
#                 (self.width - self.penalty_box_width) / 2
#             ),  # 18
#             (
#                 self.length - self.penalty_box_length,
#                 (self.width - self.goal_box_width) / 2
#             ),  # 19
#             (
#                 self.length - self.penalty_box_length,
#                 (self.width + self.goal_box_width) / 2
#             ),  # 20
#             (
#                 self.length - self.penalty_box_length,
#                 (self.width + self.penalty_box_width) / 2
#             ),  # 21
#             (self.length - self.penalty_spot_distance, self.width / 2),  # 22
#             (
#                 self.length - self.goal_box_length,
#                 (self.width - self.goal_box_width) / 2
#             ),  # 23
#             (
#                 self.length - self.goal_box_length,
#                 (self.width + self.goal_box_width) / 2
#             ),  # 24
#             (self.length, 0),  # 25
#             (self.length, (self.width - self.penalty_box_width) / 2),  # 26
#             (self.length, (self.width - self.goal_box_width) / 2),  # 27
#             (self.length, (self.width + self.goal_box_width) / 2),  # 28
#             (self.length, (self.width + self.penalty_box_width) / 2),  # 29
#             (self.length, self.width),  # 30
#             (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
#             (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
#         ]

#     edges: List[Tuple[int, int]] = field(default_factory=lambda: [
#         (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
#         (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
#         (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
#         (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
#         (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
#         (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
#     ])

#     labels: List[str] = field(default_factory=lambda: [
#         "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
#         "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
#         "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
#         "14", "19"
#     ])

#     colors: List[str] = field(default_factory=lambda: [
#         "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
#         "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
#         "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
#         "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
#         "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
#         "#00BFFF", "#00BFFF"
#     ])

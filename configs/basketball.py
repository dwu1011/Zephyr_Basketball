from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BasketballCourtConfiguration:
    width: int = 1524  # [cm]
    length: int = 2440  # [cm]
    paint_width: int = 490
    paint_length: int = 580
    arc_radius: int = 675
    center_x: int = length // 2
    center_y: int = width // 2

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
          # Center Line (01–05)
            (self.center_x, 0),                                   # 01
            (self.center_x, self.width // 4),                     # 02
            (self.center_x, self.center_y),                       # 03
            (self.center_x, 3 * self.width // 4),                 # 04
            (self.center_x, self.width),                          # 05

            # Left Paint Area (11–15)
            (0, self.center_y - self.paint_width // 2),                       # 11
            (self.paint_length // 2, self.center_y - self.paint_width // 2),  # 12
            (self.paint_length, self.center_y),                               # 13
            (self.paint_length // 2, self.center_y + self.paint_width // 2),  # 14
            (0, self.center_y + self.paint_width // 2),                       # 15

            # Left 3PT Arc (21–25) — sampled along arc
            (150, self.center_y - self.arc_radius),              # 21 (top)
            (100, self.center_y - self.arc_radius // 2),         # 22
            (0, self.center_y),                                  # 23
            (100, self.center_y + self.arc_radius // 2),         # 24
            (150, self.center_y + self.arc_radius),              # 25 (bottom)

            # Right Paint Area (31–35)
            (self.length, self.center_y - self.paint_width // 2),                           # 31
            (self.length - self.paint_length // 2, self.center_y - self.paint_width // 2),  # 32
            (self.length - self.paint_length, self.center_y),                               # 33
            (self.length - self.paint_length // 2, self.center_y + self.paint_width // 2),  # 34
            (self.length, self.center_y + self.paint_width // 2),                           # 35

            # Right 3PT Arc (41–45) — sampled along arc
            (self.length - 150, self.center_y - self.arc_radius),       # 41 (top)
            (self.length - 100, self.center_y - self.arc_radius // 2),  # 42
            (self.length, self.center_y),                               # 43
            (self.length - 100, self.center_y + self.arc_radius // 2),  # 44
            (self.length - 150, self.center_y + self.arc_radius),       # 45 (bottom)
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Center line
        (1, 2), (2, 3), (3, 4), (4, 5),

        # Left Paint
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 6),

        # Left Arc
        (11, 12), (12, 13), (13, 14), (14, 15),

        # Right Paint
        (16, 17), (17, 18), (18, 19), (19, 20), (20, 16),

        # Right Arc
        (21, 22), (22, 23), (23, 24), (24, 25),
    ])

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

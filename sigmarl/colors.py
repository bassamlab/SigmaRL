# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class RGBColor:
    def __init__(self, red, green, blue):
        self._red = red
        self._green = green
        self._blue = blue

    @property
    def value(self):
        # Convert RGB from 0-255 to 0-1 range
        return (self._red / 255, self._green / 255, self._blue / 255)


class Color:
    """Data come from https://www.color-hex.com/color-palettes/?keyword=rwth"""

    blue100 = RGBColor(0, 84, 159).value
    blue75 = RGBColor(64, 127, 183).value
    blue50 = RGBColor(142, 186, 229).value
    blue25 = RGBColor(199, 221, 242).value
    blue10 = RGBColor(232, 241, 250).value

    purple100 = RGBColor(122, 111, 172).value
    purple75 = RGBColor(155, 145, 193).value
    purple50 = RGBColor(188, 181, 215).value
    purple25 = RGBColor(222, 218, 235).value
    purple10 = RGBColor(242, 240, 247).value

    violet100 = RGBColor(97, 33, 88).value
    violet75 = RGBColor(131, 78, 117).value
    violet50 = RGBColor(168, 133, 158).value
    violet25 = RGBColor(210, 192, 205).value
    violet10 = RGBColor(237, 229, 234).value

    bordeaux100 = RGBColor(161, 16, 53).value
    bordeaux75 = RGBColor(182, 82, 86).value
    bordeaux50 = RGBColor(205, 139, 135).value
    bordeaux25 = RGBColor(229, 197, 192).value
    bordeaux10 = RGBColor(245, 232, 229).value

    red100 = RGBColor(204, 7, 30).value
    red75 = RGBColor(216, 92, 65).value
    red50 = RGBColor(230, 150, 121).value
    red25 = RGBColor(243, 205, 187).value
    red10 = RGBColor(250, 235, 227).value

    orange100 = RGBColor(246, 168, 0).value
    orange75 = RGBColor(250, 190, 80).value
    orange50 = RGBColor(253, 212, 143).value
    orange25 = RGBColor(254, 234, 201).value
    orange10 = RGBColor(255, 247, 234).value

    maygreen100 = RGBColor(189, 205, 0).value
    maygreen75 = RGBColor(208, 217, 92).value
    maygreen50 = RGBColor(224, 230, 154).value
    maygreen25 = RGBColor(240, 243, 208).value
    maygreen10 = RGBColor(249, 250, 237).value

    green100 = RGBColor(87, 171, 39).value
    green75 = RGBColor(141, 192, 96).value
    green50 = RGBColor(184, 214, 152).value
    green25 = RGBColor(221, 235, 206).value
    green10 = RGBColor(242, 247, 236).value

    turquoise100 = RGBColor(0, 152, 161).value
    turquoise75 = RGBColor(0, 177, 183).value
    turquoise50 = RGBColor(137, 204, 207).value
    turquoise25 = RGBColor(202, 231, 231).value
    turquoise10 = RGBColor(235, 246, 246).value

    petrol100 = RGBColor(0, 97, 101).value
    petrol75 = RGBColor(45, 127, 131).value
    petrol50 = RGBColor(125, 164, 167).value
    petrol25 = RGBColor(191, 208, 209).value
    petrol10 = RGBColor(230, 236, 236).value

    yellow100 = RGBColor(255, 237, 0).value
    yellow75 = RGBColor(255, 240, 85).value
    yellow50 = RGBColor(255, 245, 155).value
    yellow25 = RGBColor(255, 250, 209).value
    yellow10 = RGBColor(255, 253, 238).value

    magenta100 = RGBColor(227, 0, 102).value
    magenta75 = RGBColor(233, 96, 136).value
    magenta50 = RGBColor(241, 158, 177).value
    magenta25 = RGBColor(249, 210, 218).value
    magenta10 = RGBColor(253, 238, 240).value

    black100 = RGBColor(0, 0, 0).value
    black75 = RGBColor(100, 101, 103).value
    black50 = RGBColor(156, 158, 159).value
    black25 = RGBColor(207, 209, 210).value
    black10 = RGBColor(236, 237, 237).value


colors = [
    Color.blue100,
    Color.maygreen100,
    Color.violet100,
    Color.green100,
    Color.red100,
    Color.orange100,
    Color.purple100,
    Color.bordeaux100,
    Color.turquoise50,
    Color.petrol100,
    Color.yellow100,
    Color.magenta100,
    Color.black100,
    Color.blue50,
    Color.maygreen50,
    Color.violet50,
    Color.green50,
    Color.red50,
    Color.orange50,
    Color.purple50,
    Color.bordeaux50,
    Color.turquoise50,
    Color.petrol50,
    Color.yellow50,
    Color.magenta50,
    Color.black50,
]

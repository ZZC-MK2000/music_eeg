"""
正式实验开始前的页面, 医采
"""
from psychopy import event, core

from utils.basics.Text import *


class Start:
    """
    实验开始前的引导页面
    """
    def __init__(self, win, params):
        self.win = win
        self.params = params
    
    def main_exp(self, demo=False):
        self.win.setColor([-1, -1, -1], 'rgb')
        self.win.flip()
        title = TitleText(self.win, text="欢迎", pos=(0.0, 365/1080))
        text = Text(self.win, 
                    text="您好，欢迎参加本\"听觉脑电\"实验。\n您将收听若干个音乐片段，\n请保持放松",
                    height=50)
        hint = SpaceText(self.win, text="请按回车（Enter）键继续")

        event.clearEvents('all')
        while True:
            title.draw()
            text.draw()
            hint.draw()
            self.win.flip()
            keys = event.getKeys(['return', 'escape', 'q'])
            if 'escape' in keys or 'q' in keys:
                return False
            if 'return' in keys:
                break
        return True
        
"""
范式之间，进度提示以及结束提示
"""

from psychopy import event
from psychopy.visual import TextStim
from psychopy.visual.rect import Rect


class Join:
    
    def __init__(self, win):
        self.win = win
    
    def make_prog(self, cur_exp, total_exp):
        rect = Rect(self.win, fillColor=[-0.98, 0.33, 0.84], lineColor=[-0.98, 0.33, 0.84], pos=(-0.25*total_exp/2+0.25*cur_exp/2, -0.8), height=0.1, width=0.25*cur_exp)
        rect_outline = Rect(self.win, fillColor='gray', lineColor='black', pos=(0, -0.8), height=0.1, width=0.25*total_exp)
        rect_outline.draw()
        rect.draw()
        pro_text  = '当前实验进度' + str(cur_exp) + '/' + str(total_exp)
        progress_remind = TextStim(self.win, font="PingFang SC",text=pro_text, color=1, pos=[0, -0.8], wrapWidth=500)    
        progress_remind.draw()
        self.win.flip()
        
        event.clearEvents('all')
        while True:
            if event.getKeys('return'):
                break
    
    def end(self):
        self.win.setColor([-1, -1, -1], 'rgb')
        self.win.flip()
        
        confirm_text = TextStim(self.win, font="PingFang SC", text=u'实验结束，感谢参与', color=1)
        confirm_text.draw()
        self.win.flip()
        event.clearEvents('all')
        while True:
            keys = event.getKeys(['return', 'escape', 'q'])
            if 'escape' in keys or 'q' in keys:
                return False
            if 'return' in keys:
                break
        return True
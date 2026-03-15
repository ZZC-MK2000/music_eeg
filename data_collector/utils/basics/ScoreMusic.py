from psychopy import core, event
from psychopy.visual import TextStim, Rect

class ScoreMusic:
    def __init__(self, win):
        self.win = win

    def rate_music(self, cur_trial, total_trials, genre):
        """
        对音乐进行多维度评分
        包括: 唤醒度(arousal)、效价(valence)、喜好度(liking)
        """
        ratings = {}
        
        # 1. 唤醒度评分 (平静-兴奋)
        ratings['arousal'] = self._single_dimension_rating(
            cur_trial, total_trials,
            title="唤醒度评分",
            question="这段音乐让您感到有多兴奋？",
            left_label="非常平静",
            right_label="非常兴奋"
        )
        
        # 2. 效价评分 (负面-正面)
        ratings['valence'] = self._single_dimension_rating(
            cur_trial, total_trials,
            title="效价评分",
            question="这段音乐给您带来的感受是？",
            left_label="非常负面",
            right_label="非常正面"
        )
        
        # 3. 喜好度评分 (不喜欢-喜欢)
        ratings['liking'] = self._single_dimension_rating(
            cur_trial, total_trials,
            title="喜好度评分",
            question="您有多喜欢这段音乐？",
            left_label="非常不喜欢",
            right_label="非常喜欢"
        )
        
        return ratings
    
    def _single_dimension_rating(self, cur_trial, total_trials, title, question, left_label, right_label):
        """单维度评分量表"""
        # 进度提示
        progress_text = TextStim(
            self.win, 
            text=f'进度: {cur_trial}/{total_trials}', 
            font="PingFang SC",
            color=1, 
            pos=(-0.7, 0.9),
            height=0.04
        )
        
        # 标题
        title_text = TextStim(
            self.win, 
            text=title, 
            font="PingFang SC",
            color=1, 
            pos=(0, 0.6),
            height=0.06,
            bold=True
        )
        
        # 问题
        question_text = TextStim(
            self.win, 
            text=question, 
            font="PingFang SC",
            color=1, 
            pos=(0, 0.4),
            height=0.05
        )
        
        # 评分条
        scale_outline = Rect(
            self.win, 
            fillColor='white', 
            lineColor='black',
            pos=(0, 0), 
            height=0.1, 
            width=1.2
        )
        
        scale_fill = Rect(
            self.win, 
            fillColor=[0.35, 1, 0.2], 
            lineColor=[0.35, 1, 0.2],
            pos=(-0.6, 0), 
            height=0.1, 
            width=0
        )
        
        # 两端标签
        left_text = TextStim(
            self.win, 
            text=left_label, 
            font="PingFang SC",
            color=1, 
            pos=(-0.6, -0.2),
            height=0.04
        )
        
        right_text = TextStim(
            self.win, 
            text=right_label, 
            font="PingFang SC",
            color=1, 
            pos=(0.6, -0.2),
            height=0.04
        )
        
        # 确认提示
        confirm_text = TextStim(
            self.win, 
            text="请按左右方向键选择，按Enter键确认", 
            font="PingFang SC",
            color=1, 
            pos=(0, -0.5),
            height=0.04
        )
        
        # 初始评分值(1-9分)
        rating = 5
        max_rating = 9
        
        event.clearEvents('all')
        
        while True:
            # 更新评分条显示
            scale_fill.width = (rating / max_rating) * scale_outline.width
            scale_fill.pos = (
                scale_outline.pos[0] - scale_outline.width/2 + scale_fill.width/2,
                0
            )
            
            # 绘制所有元素
            progress_text.draw()
            title_text.draw()
            question_text.draw()
            scale_outline.draw()
            scale_fill.draw()
            left_text.draw()
            right_text.draw()
            confirm_text.draw()
            
            self.win.flip()
            core.wait(0.1)
            
            # 处理按键
            keys = event.waitKeys()
            if 'right' in keys:
                rating = min(rating + 1, max_rating)
            elif 'left' in keys:
                rating = max(rating - 1, 1)
            elif 'return' in keys:
                break
            elif 'q' in keys:
                print("用户退出实验")
                core.quit()
        
        return rating
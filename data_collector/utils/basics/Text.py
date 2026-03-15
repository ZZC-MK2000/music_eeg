from psychopy.visual import TextStim

class Text:
    def __init__(self, win,
                 text="Hello World",
                 font="PingFang SC",    # 与TextStim的区别
                 pos=(0.0, 0.0),
                 depth=0,
                 rgb=None,
                 color="#707070", # 与TextStim的区别
                 colorSpace='rgb',
                 opacity=1.0,
                 contrast=1.0,
                 units="height",    # 与TextStim的区别
                 ori=0.0,
                 height=40,  
                 antialias=True,
                 bold=False,
                 italic=False,
                 alignHoriz=None,
                 alignVert=None,
                 alignText='center',
                 anchorHoriz='center',
                 anchorVert='center',
                 fontFiles=(),
                 wrapWidth=None,
                 flipHoriz=False,
                 flipVert=False,
                 languageStyle='LTR',
                 name=None,
                 autoLog=None,
                 autoDraw=False):

        self.win = win
        self.pos = pos
        self.text = TextStim(win=win, text=text, font=font, pos=pos, depth=depth, rgb=rgb, color=color,
                             colorSpace=colorSpace, opacity=opacity, contrast=contrast, units=units,
                             ori=ori, height=height/self.win.size[1], antialias=antialias, bold=bold, italic=italic,
                             alignHoriz=alignHoriz, alignVert=alignVert, alignText=alignText,
                             anchorHoriz=anchorHoriz, anchorVert=anchorVert, fontFiles=fontFiles,
                             wrapWidth=wrapWidth, flipHoriz=flipHoriz, flipVert=flipVert,
                             languageStyle=languageStyle, name=name, autoLog=autoLog)

    def setText(self, text, log=False):
        self.text.setText(text=text)
    
    def setPos(self, newPos, operation='', log=None):
        self.text.setPos(newPos=newPos)
        
    def draw(self):
        self.text.draw()


class SpaceText(Text):
    def __init__(self, win,
                 text="请按下“空格”键继续",
                 font="PingFang SC",    # 与TextStim的区别
                 pos=(0.0, -150/540),    # 与Text区别
                 depth=0,
                 rgb=None,
                 color="#4D4D4D",    # 与Text区别
                 colorSpace='rgb',
                 opacity=1.0,
                 contrast=1.0,
                 units="height",
                 ori=0.0,
                 height=40,     # 与Text区别
                 antialias=True,
                 bold=False,
                 italic=False,
                 alignHoriz=None,
                 alignVert=None,
                 alignText='center',
                 anchorHoriz='center',
                 anchorVert='center',
                 fontFiles=(),
                 wrapWidth=None,
                 flipHoriz=False,
                 flipVert=False,
                 languageStyle='LTR',
                 name=None,
                 autoLog=None,
                 autoDraw=False 
                ):
        
        self.win = win
        self.text = TextStim(win=win, text=text, font=font, pos=pos, depth=depth, rgb=rgb, color=color,
                             colorSpace=colorSpace, opacity=opacity, contrast=contrast, units=units,
                             ori=ori, height=height/self.win.size[1], antialias=antialias, bold=bold, italic=italic,
                             alignHoriz=alignHoriz, alignVert=alignVert, alignText=alignText,
                             anchorHoriz=anchorHoriz, anchorVert=anchorVert, fontFiles=fontFiles,
                             wrapWidth=wrapWidth, flipHoriz=flipHoriz, flipVert=flipVert,
                             languageStyle=languageStyle, name=name, autoLog=autoLog)


class TitleText(Text):
    def __init__(self, win,
                 text="Hello World",
                 font="PingFang SC",    # 与TextStim的区别
                 pos=(0.0, 175/1080),    # 与Text区别
                 depth=0,
                 rgb=None,
                 color="#707070",    # 与TextStim的区别
                 colorSpace='rgb',
                 opacity=1.0,
                 contrast=1.0,
                 units="height",
                 ori=0.0,
                 height=50,
                 antialias=True,
                 bold=False,
                 italic=False,
                 alignHoriz=None,
                 alignVert=None,
                 alignText='center',
                 anchorHoriz='center',
                 anchorVert='center',
                 fontFiles=(),
                 wrapWidth=None,
                 flipHoriz=False,
                 flipVert=False,
                 languageStyle='LTR',
                 name=None,
                 autoLog=None,
                 autoDraw=False
                ):
        
        self.win = win
        self.text = TextStim(win=win, text=text, font=font, pos=pos, depth=depth, rgb=rgb, color=color,
                             colorSpace=colorSpace, opacity=opacity, contrast=contrast, units=units,
                             ori=ori, height=height/self.win.size[1], antialias=antialias, bold=bold, italic=italic,
                             alignHoriz=alignHoriz, alignVert=alignVert, alignText=alignText,
                             anchorHoriz=anchorHoriz, anchorVert=anchorVert, fontFiles=fontFiles,
                             wrapWidth=wrapWidth, flipHoriz=flipHoriz, flipVert=flipVert,
                             languageStyle=languageStyle, name=name, autoLog=autoLog)        
    

class SubTitleText(Text):
    def __init__(self, win,
                 text="Hello World",
                 font="PingFang SC",    # 与TextStim的区别
                 pos=(0.0, 70/1080),    # 与Text区别
                 depth=0,
                 rgb=None,
                 color="#707070",    # 与TextStim的区别
                 colorSpace='rgb',
                 opacity=1.0,
                 contrast=1.0,
                 units="height",
                 ori=0.0,
                 height=40,
                 antialias=True,
                 bold=False,
                 italic=False,
                 alignHoriz=None,
                 alignVert=None,
                 alignText='center',
                 anchorHoriz='center',
                 anchorVert='center',
                 fontFiles=(),
                 wrapWidth=None,
                 flipHoriz=False,
                 flipVert=False,
                 languageStyle='LTR',
                 name=None,
                 autoLog=None,
                 autoDraw=False
                ):
        
        self.win = win
        self.text = TextStim(win=win, text=text, font=font, pos=pos, depth=depth, rgb=rgb, color=color,
                             colorSpace=colorSpace, opacity=opacity, contrast=contrast, units=units,
                             ori=ori, height=height/self.win.size[1], antialias=antialias, bold=bold, italic=italic,
                             alignHoriz=alignHoriz, alignVert=alignVert, alignText=alignText,
                             anchorHoriz=anchorHoriz, anchorVert=anchorVert, fontFiles=fontFiles,
                             wrapWidth=wrapWidth, flipHoriz=flipHoriz, flipVert=flipVert,
                             languageStyle=languageStyle, name=name, autoLog=autoLog)
        
import os
from utils.bootstrap import configure_psychopy

configure_psychopy()

from psychopy import visual, core
from utils.Start import Start
from utils.MusicPlayer import MusicPlayer

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 实验参数设置
    params = {
        'base_dir': base_dir,
        # 音乐文件路径，按类型分组
        'music_files': {
            'classical': [
                os.path.join(base_dir, 'materials', 'music', 'classical.mp3')
            ],
            'rock': [
                os.path.join(base_dir, 'materials', 'music', 'rock.mp3')
            ],
            'jazz': [
                os.path.join(base_dir, 'materials', 'music', 'jazz.mp3')
            ],
            'ambient': [
                os.path.join(base_dir, 'materials', 'music', 'ambient.mp3')
            ]
        },
        # 每种音乐类型的trigger索引
        'trigger_index': {
            'classical': 1,
            'rock': 2,
            'jazz': 3,
            'ambient': 4
        },
        # 每个音乐片段播放时长(秒)，0表示播放完整首
        'music_duration': 30,
        # 试次间休息时长(秒)
        'rest_duration': 10,
        # 总实验块数
        'total_blocks': 1,
        # 每块中每种音乐类型的试次数
        'trials_per_genre': 1
    }
    
    # 创建窗口
    
    win = visual.Window(
        size=(1200, 800),
        fullscr=False,
        screen=0,
        winType='pyglet',
        allowGUI=False,
        allowStencil=False,
        monitor='testMonitor',
        color=[0, 0, 0],
        colorSpace='rgb',
        blendMode='avg',
        useFBO=True,
        units='height'
    )
    
    # 显示开始界面
    start = Start(win, params)
    if not start.main_exp():
        win.close()
        core.quit()
        return
    
    # 运行音乐实验
    music_player = MusicPlayer(win, params)
    if not music_player.main_exp():
        win.close()
        core.quit()
        return
    
    # 实验结束
    win.close()
    core.quit()

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('./save/', exist_ok=True)
    os.makedirs('./materials/music/', exist_ok=True)
    main()
"""Simple MusicPlayer for EEG experiment."""

import os
import sys
import csv
import random
import time
from datetime import datetime

# Ensure project root is on sys.path when running this file directly
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.bootstrap import configure_psychopy

configure_psychopy()

from psychopy import event, core, sound, constants
from psychopy.visual import TextStim

from utils.Join import Join
from utils.basics.ScoreMusic import ScoreMusic
from utils.Trigger import ActiviewTrigger


class MusicPlayer:
    def __init__(self, win, params):
        self.win = win
        self.params = params
        self.join = Join(win)
        self.score = ScoreMusic(win)

        self.trigger = ActiviewTrigger(port='COM5')
        self.trigger_connected = self.trigger.connect()
        if not self.trigger_connected:
            print("警告: Trigger连接失败，将仅记录本地时间戳")

        self.timestamp_file = './save/music_timestamps.csv'
        os.makedirs('./save/', exist_ok=True)
        with open(self.timestamp_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Event', 'Timestamp', 'Unix_Timestamp', 'Trigger_Index', 'Trigger_Sent', 'Music_Type', 'Music_File'])

        self.prepare_experiment_sequence()

    def _abs_path(self, path):
        if os.path.isabs(path):
            return path
        base_dir = self.params.get('base_dir', '.')
        return os.path.abspath(os.path.join(base_dir, path))

    def prepare_experiment_sequence(self):
        self.experiment_sequence = []
        music_types = list(self.params['music_files'].keys())

        for block in range(self.params['total_blocks']):
            for _ in range(self.params['trials_per_genre']):
                shuffled = random.sample(music_types, len(music_types))
                for genre in shuffled:
                    music_file = random.choice(self.params['music_files'][genre])
                    self.experiment_sequence.append({
                        'block': block + 1,
                        'genre': genre,
                        'file': self._abs_path(music_file),
                        'trigger': self.params['trigger_index'][genre]
                    })

        with open('./save/experiment_sequence.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Block', 'Genre', 'Music_File', 'Trigger_Index'])
            for trial in self.experiment_sequence:
                writer.writerow([trial['block'], trial['genre'], trial['file'], trial['trigger']])

    def record_timestamp(self, event_name, trigger_num=None, music_type=None, music_file=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        unix_timestamp = time.time()
        trigger_sent = False

        if trigger_num is not None and self.trigger_connected:
            try:
                trigger_sent = self.trigger.send_trigger(trigger_num, duration_ms=50)
            except Exception as e:
                print(f'{event_name}: {timestamp} - Trigger发送异常: {e}')
                trigger_sent = False

        print(f'{event_name}: {timestamp}')

        with open(self.timestamp_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([event_name, timestamp, unix_timestamp, trigger_num, trigger_sent, music_type, music_file])

    def _make_text(self, text, pos=(0, 0), height=0.05):
        return TextStim(self.win, text=text, font="PingFang SC", color=1, pos=pos, height=height)

    def _play_with_psychopy(self, music_file, duration=0, trial_num=None, total_trials=None, genre=None):
        try:
            music = sound.Sound(music_file, stereo=True, audioLib='pygame')
        except Exception:
            music = sound.Sound(music_file, stereo=True)
        music.setVolume(0.8)
        try:
            file_duration = music.getDuration()
        except Exception:
            file_duration = None
        audio_lib = getattr(music, 'audioLib', None)
        print(f"音频后端: {audio_lib}")
        if file_duration:
            print(f"音频文件时长(秒): {file_duration:.2f}")
        music.play()

        event.clearEvents('all')
        clock = core.Clock()
        playing_text = self._make_text("正在播放音乐...")
        remaining_text = self._make_text("", pos=(0, -0.5), height=0.04)
        hint_text = self._make_text("空格=跳过  Q/Esc=退出", pos=(0, -0.8), height=0.035)
        status_parts = ["阶段: 音乐播放"]
        if genre:
            status_parts.append(f"类型: {genre}")
        if trial_num and total_trials:
            status_parts.append(f"进度: {trial_num}/{total_trials}")
        status_text = self._make_text(" | ".join(status_parts), pos=(0, 0.8), height=0.035)

        target_duration = duration if duration and duration > 0 else None
        display_duration = target_duration or file_duration

        while True:
            elapsed = clock.getTime()
            playing_text.draw()
            if display_duration:
                remaining = max(0, display_duration - elapsed)
                remaining_text.setText(f"剩余时间: {int(remaining + 0.999)}秒")
            else:
                remaining_text.setText("播放中...")
            remaining_text.draw()
            hint_text.draw()
            status_text.draw()
            self.win.flip()

            keys = event.getKeys(keyList=['space', 'q', 'escape'])
            if (target_duration and elapsed >= target_duration) or ('space' in keys):
                music.stop()
                return True
            if 'q' in keys or 'escape' in keys:
                music.stop()
                print("用户退出实验")
                return False

            if (not target_duration) and music.status == constants.FINISHED:
                music.stop()
                return True

            if target_duration and music.status == constants.FINISHED and elapsed < target_duration - 0.5:
                print("警告: 音频提前结束，可能是解码或设备问题")

            core.wait(0.02)

    def play_music(self, music_file, duration=0, trial_num=None, total_trials=None, genre=None):
        if not os.path.exists(music_file):
            print(f"音乐文件不存在: {music_file}")
            return False

        ext = os.path.splitext(music_file)[1].lower()
        if ext not in ['.mp3', '.wav']:
            print(f"不支持的音频格式: {ext}")
            return False

        try:
            loading_text = self._make_text("正在加载音乐...")
            loading_text.draw()
            self.win.flip()
            core.wait(0.1)
        except Exception as e:
            print(f"播放音乐时出错: {e}")
            return False
        return self._play_with_psychopy(music_file, duration, trial_num, total_trials, genre)

    def rest_period(self, duration, trial_num, total_trials):
        rest_text = self._make_text("休息中...")
        remaining_text = self._make_text("", pos=(0, -0.5), height=0.04)
        progress_text = self._make_text(f'进度: {trial_num}/{total_trials}', pos=(-0.7, 0.9), height=0.04)
        hint_text = self._make_text("空格=继续  Q/Esc=退出", pos=(0, -0.8), height=0.035)
        status_text = self._make_text("阶段: 休息", pos=(0, 0.8), height=0.035)

        start_time = core.getTime()
        self.record_timestamp("休息开始")

        while True:
            remaining = max(0, duration - (core.getTime() - start_time))
            rest_text.draw()
            remaining_text.setText(f"剩余休息时间: {int(remaining)}秒")
            remaining_text.draw()
            progress_text.draw()
            hint_text.draw()
            status_text.draw()
            self.win.flip()

            if remaining <= 0 or event.getKeys('space'):
                break
            if event.getKeys(['q', 'escape']):
                print("用户退出实验")
                return False
            core.wait(0.1)

        self.record_timestamp("休息结束")
        return True

    def main_exp(self):
        total_trials = len(self.experiment_sequence)
        trial_num = 0

        self.record_timestamp("实验开始", 1)

        instructions = self._make_text(
            "接下来您将听到不同类型的音乐。\n请放松并专注地聆听每一段音乐。\n每段音乐结束后，我们会请您对音乐感受进行评分。\n\n按回车键开始",
            height=0.05
        )
        instructions.draw()
        self.win.flip()
        keys = event.waitKeys(keyList=['return', 'escape', 'q'])
        if 'escape' in keys or 'q' in keys:
            return False

        for trial in self.experiment_sequence:
            trial_num += 1
            block_num = trial['block']
            genre = trial['genre']
            music_file = trial['file']
            trigger = trial['trigger']

            trial_info = self._make_text(f"即将播放: {genre}\n\n按回车键开始")
            progress_text = self._make_text(f'进度: {trial_num}/{total_trials} (第{block_num}块)', pos=(-0.7, 0.9), height=0.04)
            hint_text = self._make_text("Enter=开始  Q/Esc=退出", pos=(0, -0.8), height=0.035)
            status_text = self._make_text("阶段: 播放准备", pos=(0, 0.8), height=0.035)
            trial_info.draw()
            progress_text.draw()
            hint_text.draw()
            status_text.draw()
            self.win.flip()
            keys = event.waitKeys(keyList=['return', 'escape', 'q'])
            if 'escape' in keys or 'q' in keys:
                return False

            self.record_timestamp("音乐播放开始", trigger, genre, music_file)
            if not self.play_music(music_file, self.params['music_duration'], trial_num, total_trials, genre):
                return False
            self.record_timestamp("音乐播放结束", trigger, genre, music_file)

            rating = self.score.rate_music(trial_num, total_trials, genre)
            with open('./save/music_ratings.csv', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if trial_num == 1:
                    writer.writerow(['Trial_Num', 'Block', 'Genre', 'Music_File', 'Arousal', 'Valence', 'Liking'])
                writer.writerow([trial_num, block_num, genre, music_file, rating['arousal'], rating['valence'], rating['liking']])

            if trial_num < total_trials:
                if not self.rest_period(self.params['rest_duration'], trial_num, total_trials):
                    return False

        self.record_timestamp("实验结束", 2)
        return self.join.end()


def _demo_params():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return {
        'base_dir': base_dir,
        'music_files': {
            'test': [os.path.join(base_dir, 'materials', 'music', 'classical.mp3')]
        },
        'trigger_index': {'test': 1},
        'music_duration': 3,
        'rest_duration': 1,
        'total_blocks': 1,
        'trials_per_genre': 1
    }


def _run_smoke_test():
    from psychopy import visual

    win = visual.Window(size=(800, 600), fullscr=False, color=[0, 0, 0], units='height')
    try:
        player = MusicPlayer(win, _demo_params())
        test_file = player.params['music_files']['test'][0]
        player.record_timestamp("测试开始", trigger_num=1, music_type='test', music_file=test_file)
        player.play_music(test_file, duration=3)
        player.record_timestamp("测试结束", trigger_num=2, music_type='test', music_file=test_file)
    finally:
        win.close()
        core.quit()


if __name__ == "__main__":
    _run_smoke_test()
import numpy as np
import time
import cv2
import os

class Android(object):
    def __init__(self, cache='./cache', debug=False):
        if not os.path.exists(cache):
            os.makedirs(cache)
        self.cache_dir = cache
        self.cache_name = os.path.join(cache, 'state.png')
        self.debug = debug
        self.player = cv2.imread('./resource/player.png')

    def get_screen(self):
        os.system('adb shell screencap -p /sdcard/tmp.png')
        os.system('adb pull /sdcard/tmp.png {}'.format(self.cache_name))

        img = cv2.imread(self.cache_name)
        h, w = img.shape[:2]
        img = cv2.resize(img, (720, int(h*720./w)))
        if img.shape[0] > 1280:
            img = img[img.shape[0]-1280:]
        elif img.shape[0] < 1280:
            img = np.concatenate([img[:1280-img.shape[0]][::-1], img], axis=0)
        assert img.shape == (1280, 720, 3), img.shape
        return img

    def jump(self, pt1, pt2):
        dist = np.linalg.norm(pt1 - pt2)
        time = int(dist**0.88 * 4.38)
        os.system('adb shell input swipe {} {} {} {} {}'.format(
            *(list(np.random.randint(300, 400, (4,))) + [time])))

    def get_player_pos(self, img):
        score_map = cv2.matchTemplate(img, self.player, cv2.TM_CCORR_NORMED)
        index = score_map.argmax()
        row = index // score_map.shape[1]
        col = index % score_map.shape[1]
        pos = np.array([col + self.player.shape[1]*0.5, row+self.player.shape[0]*0.9]).astype(np.int32)
        if self.debug:
            cv2.circle(img, tuple([int(x) for x in pos]), 5, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(self.cache_dir, 'player_pos.png'), img)
        return pos

    def get_target_pos(self, img, ppos, th=10):
        begin = ppos[1] - 400
        gray_img = [cv2.cvtColor(img[begin:], cv2.COLOR_BGR2GRAY)] +\
            [img[begin:, :, i] for i in range(3)]
        edge_img = np.array([cv2.Canny(x, th, 2*th) for x in gray_img])
        edge_img = (edge_img.astype(np.int32).sum(axis=0) > 0).astype(np.uint8) * 255

        mask = np.array([
            ppos[0] - self.player.shape[1]*0.5,
            ppos[0] + self.player.shape[1]*0.5,
            ppos[1] - self.player.shape[0]*0.9 - begin,
            ppos[1] + self.player.shape[0]*0.1 - begin]).astype(np.int32)
        edge_img[mask[2]:mask[3], mask[0]:mask[1]] = 0
        
        ends = []
        last_mean = None
        r = [None, None, None]
        for i, line in enumerate(edge_img):
            if line.max() == 0:
                begin += 1
                continue
            left = np.argmax(line)
            right = line.size - 1 - np.argmax(line[::-1])
            ends.append([left, right])
            mean = 0.5 * (left+right)
            if last_mean is None:
                last_mean = mean
            else:
                if abs(mean - last_mean) >= 0.9:
                    break
                last_mean = last_mean * 0.95 + mean * 0.05

            r = r[1:] + [left]
            if r[0] is not None:
                if abs(r[-1] - r[0]) <= 1.0:
                    break

        ends = np.array(ends)
        length = ends[:, 1] - ends[:, 0]
        pos = np.array([ends.mean(), max(length.argmax(), 40)+begin]).astype(np.int32)

        if self.debug:
            img[begin] = np.array([0, 0, 255])
            cv2.circle(img, tuple(ppos), 5, (0, 0, 255), -1)
            cv2.circle(img, tuple(pos), 5, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(self.cache_dir, 'target_pos.png'), edge_img)
            cv2.imwrite(os.path.join(self.cache_dir, 'target_pos2.png'), img)
        return pos


    def run_once(self):
        img = self.get_screen()
        pt1 = self.get_player_pos(img.copy())
        pt2 = self.get_target_pos(img.copy(), pt1)
        self.jump(pt1, pt2)
        time.sleep(1.3)

    def run(self):
        while True: self.run_once()

player = Android(debug=True)
player.run()

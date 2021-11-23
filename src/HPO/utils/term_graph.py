"""Draw a live bar graph of frequency data (from Arduino FHT)."""

import collections
import curses
import math
import serial  # http://pyserial.sourceforge.net/
import time

_SERIAL_DEVICE = '/dev/tty.usbmodemfa141'
_GRAPH_MAX_Y = 100
_FAKE_DATA = True
_SMOOTHING = None if _FAKE_DATA else 10

_SERIAL_BAUD = 115200
_FHT_START_BYTE = 255


class CursesBarGraph:
  def __init__(self):
    self._window = None
    self._max = _GRAPH_MAX_Y

  def __enter__(self):
    self._window = curses.initscr()
    return self
  def set_max(self, val):
    self._max = val
  def __exit__(self, exc_type, exc_value, traceback):
    curses.endwin()

  def Update(self, values):
    assert self._window
    h, w = self._window.getmaxyx()
    per_bucket = max(1, math.ceil(float(len(values)) / (w - 1)))
    self._window.erase()
    for column_num, v in enumerate(_AveragedChunks(values, per_bucket)):
      assert column_num < w
      self._DrawBar(column_num, v, h)

    self._DrawAxisLabels(h, w, column_num, len(values))
    self._window.refresh()

  def _DrawBar(self, column_num, value, h):
    bar_len = max(0, min(h - 1, int(h * (value / self._max))))
    # vline draws from the starting coordinate towards positive y (down).
    self._window.vline((h - 1) - bar_len, column_num, ord('|'), bar_len)

  def _DrawAxisLabels(self, h, w, max_column, num_values):
    self._window.addstr(0, 0, str(self._max))
    self._window.addstr(h - 1, 0, str(0))
    max_column_str = str(num_values)
    self._window.addstr(
        h - 1,
        min(max_column, w - (len(max_column_str) + 1)),
        max_column_str)


def _AveragedChunks(iterable, n):
  summed_v = 0
  summed_count = 0
  for v in iterable:
    summed_v += v
    summed_count += 1
    if summed_count >= n:
      yield float(summed_v) / summed_count
      summed_v = 0
      summed_count = 0
  if summed_count > 0:
    yield float(summed_v) / summed_count


global _t
_t = 0.0
def MakeUpValues():
  time.sleep(0.05)
  global _t
  _t += .05
  values = []
  for i in range(200):
    values.append(_GRAPH_MAX_Y * (math.sin(_t + i/20.0) + 1) / 2.0)
  return values


def ReadValues(ser):
  values = []
  c = ord(ser.read())
  while c != _FHT_START_BYTE:
    values.append(c)
    c = ord(ser.read())
  return values


class Smoother:
  def __init__(self, window):
    self._n = window
    self._history = collections.defaultdict(lambda: list())

  def Smooth(self, values):
    smoothed = []
    for i, v in enumerate(values):
      history = self._history[i]
      history.append(v)
      if len(history) > self._n:
        history = history[-self._n:]
      smoothed.append(float(sum(history)) / self._n)
      self._history[i] = history
    return smoothed

def print_graph(values : list):
  smoother = Smoother(_SMOOTHING) if _SMOOTHING else None
  with CursesBarGraph() as bar_graph:
    try:
      values.append(values[-1]+1)
      bar_graph.set_max( max(values))
      if _SMOOTHING:
        values = smoother.Smooth(values)
      bar_graph.Update(values)
    except KeyboardInterrupt:
      pass



if __name__ == '__main__':
  print_graph([1,2,4,6,8,16,32])
  exit()




  values = [0]
  smoother = Smoother(_SMOOTHING) if _SMOOTHING else None
  with CursesBarGraph() as bar_graph:
    try:
      while True:
        values.append(values[-1]+1)
        bar_graph.set_max( max(values))
        if _SMOOTHING:
          values = smoother.Smooth(values)
        bar_graph.Update(values)
    except KeyboardInterrupt:
      pass

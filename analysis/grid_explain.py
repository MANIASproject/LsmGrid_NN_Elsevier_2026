import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grids import Partition, SpaceType, Splits, Cell
from utils import plot_grid2D
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager


# ==================
#    NICE FONTS
# ==================
font_manager.fontManager.addfont("fonts/Nunito-Regular.ttf")
font_manager.fontManager.addfont("fonts/Nunito-Bold.ttf")
plt.rcParams["font.family"] = "Nunito"
plt.rcParams["pdf.fonttype"] = 42


# ==================
#    PLOT ELEMENTS
# ==================
cell = Cell(t_lims=(0,1), d2_lims=(0,9))

p = Partition( SpaceType.D2 )
p.assign( Splits.T, 0.33, children_is_null=True )
p.left_child.assign( Splits.D2, 0.44, children_is_null=True )
p.right_child.assign( Splits.D2, 0.55, children_is_null=True )
p.right_child.left_child.assign( Splits.T, 0.5, children_is_null=True )


plot_grid2D( p.get_grid(cell), color='black' )
plt.yticks([i for i in range(0,10)])
plt.xticks([i/10 for i in range(0,11)])
plt.ylabel('Input neuron index (d1)')
plt.xlabel('time (t)')
plt.grid()


points = [(0.1,5),
          (0.2,7),
          (0.2,8),
          (0.25,6),
          (0.4,6),
          (0.4,7),
          (0.45,7),
          (0.45,8),
          (0.6,6),
          (0.65,7),
          (0.8,8),
          (0.9,7),
          (0.5,3),
          (0.6,2),
          ]
t,ch = zip(*points)
plt.scatter(t,ch, c='red', marker="*", s=80, zorder=2)

rec1 = Rectangle((0,3.5),0.33,5, color='#fdf3e3ff')
rec2 = Rectangle((0,-0.5),0.33,4, color='#f2f8eaff')
rec3 = Rectangle((0.33,4.5),0.67,4, color='#bcf0fcff')
rec4 = Rectangle((0.33,-0.5),0.33,5, color='#fbe8e4ff')
rec5 = Rectangle((0.66,-0.5),0.34,5, color='#e9c3f7ff')

plt.gca().add_patch(rec1)
plt.gca().add_patch(rec2)
plt.gca().add_patch(rec3)
plt.gca().add_patch(rec4)
plt.gca().add_patch(rec5)


# ==================
#       SAVE
# ==================
plt.savefig('analysis/outputs/grid_explain.svg', bbox_inches="tight")

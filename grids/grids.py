from enum import Enum
from collections.abc import Iterable

# ==================================
#         ENUMS DEFINITIONS
# ==================================
class SpaceType(Enum):
    """ Partition type """
    D2 = 1
    D3 = 2

class Splits(Enum):
    T     = 1
    D2    = 2
    D3    = 3
    NULL  = 4
    UNDEF = 5

# ==================================
#               CELL
# ==================================
class Cell:
    def __init__(self, t_lims:tuple[float,float], d2_lims:tuple[int,int], d3_lims:tuple[int,int]|None=None):
        assert isinstance(t_lims,Iterable) and len(t_lims)==2   ,"t_lims must be a tuple of 2 elements"
        assert isinstance(d2_lims,Iterable) and len(d2_lims)==2 ,"d2_lims must be a tuple of 2 elements"
        if d3_lims is not None:
            assert isinstance(d3_lims,Iterable) and len(d3_lims)==2 ,"d3_lims must be a tuple of 2 elements"

        if d3_lims is None: self.cell_type = SpaceType.D2
        else              : self.cell_type = SpaceType.D3 

        self.t_lims     = t_lims
        self.d2_lims    = d2_lims
        self.d3_lims    = d3_lims

    def split(self, split_type:Splits, split_p:float):
        if split_type!=Splits.T and split_type!=Splits.D2 and split_type!=Splits.D3:
            raise Exception('split_type must be a T, D2 or D3 SPlit')

        if split_type == Splits.D3 and self.cell_type==SpaceType.D2:
            raise Exception('3D split not supported for 2D cells')

        if split_type == Splits.T:
            start,end = self.t_lims
            mid       = start + (end-start)*split_p
            cell1     = Cell( t_lims=(start,mid), d2_lims=self.d2_lims, d3_lims=self.d3_lims )         
            cell2     = Cell( t_lims=(mid,end)  , d2_lims=self.d2_lims, d3_lims=self.d3_lims )         
        elif split_type == Splits.D2:
            start,end = self.d2_lims
            mid       = start + (end-start)*split_p
            cell1     = Cell( t_lims=self.t_lims, d2_lims=(start,mid), d3_lims=self.d3_lims ) 
            cell2     = Cell( t_lims=self.t_lims, d2_lims=(mid,end)  , d3_lims=self.d3_lims ) 
        elif split_type == Splits.D3:
            start,end = self.d3_lims
            mid       = start + (end-start)*split_p
            cell1     = Cell( t_lims=self.t_lims, d2_lims=self.d2_lims, d3_lims=(start,mid) ) 
            cell2     = Cell( t_lims=self.t_lims, d2_lims=self.d2_lims, d3_lims=(mid,end) ) 
        return cell1,cell2
    
    def round(self,precision):
        self.t_lims = ( round(self.t_lims[0],precision), round(self.t_lims[1],precision) )
        self.d2_lims = ( int(self.d2_lims[0]), int(self.d2_lims[1]) )
        if self.cell_type==SpaceType.D3:
            self.d3_lims = ( int(self.d3_lims[0]), int(self.d3_lims[1]) )
        return self
    
    def to_dict(self):
        if self.cell_type==SpaceType.D2     :    return {'t_lims':self.t_lims, 'd2_lims':self.d2_lims}
        elif self.cell_type==SpaceType.D3   :    return {'t_lims':self.t_lims, 'd2_lims':self.d2_lims, 'd3_lims':self.d3_lims}

    
    def __str__(self):
        if self.cell_type==SpaceType.D2:
            return f'Cell2D(t_lims={self.t_lims},d2_lims={self.d2_lims})'
        elif self.cell_type==SpaceType.D3:
            return f'Cell3D(t_lims={self.t_lims},d2_lims={self.d2_lims},d3_lims={self.d3_lims})'
    def __repr__(self):
        return self.__str__()

# ==================================
#           PARTITIONS
# ==================================

def check_cells_len(cell:Cell):
    if cell.d2_lims[1]-cell.d2_lims[0] < 1:
        return False
    if cell.cell_type == SpaceType.D3:
        if cell.d3_lims[1] - cell.d3_lims[0] < 1:
            return False
    return True


class Partition:
    def __init__(self, ptype:SpaceType):
        assert type(ptype) == SpaceType
        self.ptype = ptype
        self.right_child = None
        self.left_child  = None
        self.split_type  = Splits.UNDEF
        self.split_p     = None

    def assign(self, split_type:Splits, split_p:float|None=None, children_is_null=False ):
        if type(split_type)!=Splits:
            raise Exception('split type must be member of class Splits')

        if split_type==Splits.D3 and self.ptype==SpaceType.D2:
            raise Exception('Is not possible to assign a 3D split to a 2D partition.')

        if (split_type!=Splits.NULL and split_type!=Splits.UNDEF) and split_p is None:
            raise Exception("You must specify a split proportion with T, D2 and D3 splits")
        
        self.split_type = split_type
        self.split_p    = split_p
        self.left_child = Partition(self.ptype)
        self.right_child = Partition(self.ptype)

        if children_is_null:
            self.left_child.assign( Splits.NULL )
            self.right_child.assign( Splits.NULL )

        return self
    
    def get_undef(self):
        if self.split_type == Splits.UNDEF: 
            return [self]

        elif self.split_type == Splits.NULL:
            return []
        
        else:
            return self.left_child.get_undef() + self.right_child.get_undef()
        
    def find_leaves(self,level=1):
        if   self.split_type==Splits.UNDEF:   raise Exception("Cannot find leaves in partitions with UNDEF nodes")
        elif self.split_type==Splits.NULL:    return [(level,self)]
        else:
            return self.left_child.find_leaves(level+1) + self.right_child.find_leaves(level+1)
        
    def get_grid(self,cell:Cell):
        assert type(cell)==Cell, 'cell must be of type Cell'

        if self.split_type == Splits.UNDEF:
            raise Exception('Cannot generate a grid in a partition with UNDEF splits')

        elif self.split_type == Splits.NULL:
            return [cell]
        else:
            cell1,cell2 = cell.split( self.split_type, self.split_p )
            if not check_cells_len(cell1) or not check_cells_len(cell2):
                return [cell]
            return self.left_child.get_grid( cell1 ) + self.right_child.get_grid( cell2 )
    
    def __str__(self):
        if   self.split_type == Splits.T:  return f'P(T, {self.split_p})'  
        elif self.split_type == Splits.D2: return f'P(2D, {self.split_p})'  
        elif self.split_type == Splits.D3: return f'P(3D, {self.split_p})'  
        elif self.split_type == Splits.NULL: return f'P(NULL)'  
        elif self.split_type == Splits.UNDEF: return f'P(UNDEF)'
        else:
            raise Exception('Is not possible to print partition. split_type must be member of class Splits')

    def __repr__(self):
        return self.__str__()
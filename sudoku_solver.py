"""
数独求解器 (Sudoku Solver)
使用回溯算法解决数独问题
"""


class SudokuSolver:
    """数独求解器类"""
    
    def __init__(self, board):
        """
        初始化数独求解器
        
        Args:
            board: 9x9的数独棋盘，0表示空格
        """
        self.board = [row[:] for row in board]  # 深拷贝棋盘
        self.size = 9
    
    def is_valid(self, row, col, num):
        """
        检查在指定位置放置数字是否有效
        
        Args:
            row: 行索引
            col: 列索引
            num: 要放置的数字 (1-9)
        
        Returns:
            bool: 如果放置有效返回True，否则返回False
        """
        # 检查行
        for c in range(self.size):
            if self.board[row][c] == num:
                return False
        
        # 检查列
        for r in range(self.size):
            if self.board[r][col] == num:
                return False
        
        # 检查3x3宫格
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if self.board[r][c] == num:
                    return False
        
        return True
    
    def find_empty(self):
        """
        查找下一个空格位置
        
        Returns:
            tuple: (row, col) 如果找到空格，否则返回 None
        """
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    return (r, c)
        return None
    
    def solve(self):
        """
        使用回溯算法解决数独
        
        Returns:
            bool: 如果找到解返回True，否则返回False
        """
        empty = self.find_empty()
        
        # 如果没有空格，说明已经解决
        if empty is None:
            return True
        
        row, col = empty
        
        # 尝试数字1-9
        for num in range(1, 10):
            if self.is_valid(row, col, num):
                # 放置数字
                self.board[row][col] = num
                
                # 递归尝试解决剩余部分
                if self.solve():
                    return True
                
                # 如果当前数字导致无解，回溯
                self.board[row][col] = 0
        
        # 如果所有数字都无效，返回False触发回溯
        return False
    
    def print_board(self):
        """打印数独棋盘"""
        print("\n" + "=" * 25)
        for i in range(self.size):
            if i % 3 == 0 and i != 0:
                print("-" * 25)
            
            row_str = ""
            for j in range(self.size):
                if j % 3 == 0 and j != 0:
                    row_str += " | "
                
                if self.board[i][j] == 0:
                    row_str += ". "
                else:
                    row_str += str(self.board[i][j]) + " "
            
            print(row_str)
        print("=" * 25 + "\n")
    
    def get_solution(self):
        """
        获取解决方案
        
        Returns:
            list: 解决后的9x9棋盘
        """
        return [row[:] for row in self.board]


def create_example_board():
    """
    创建一个示例数独题目（中等难度）
    
    Returns:
        list: 9x9的数独棋盘
    """
    return [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]


def input_board():
    """
    从用户输入读取数独棋盘
    
    Returns:
        list: 9x9的数独棋盘
    """
    print("请输入数独棋盘（9行，每行9个数字，用空格分隔，0表示空格）：")
    board = []
    for i in range(9):
        while True:
            try:
                row = input(f"第 {i+1} 行: ").strip().split()
                if len(row) != 9:
                    print("错误：每行必须包含9个数字，请重新输入")
                    continue
                
                row = [int(x) for x in row]
                if any(x < 0 or x > 9 for x in row):
                    print("错误：数字必须在0-9之间，请重新输入")
                    continue
                
                board.append(row)
                break
            except ValueError:
                print("错误：请输入有效的数字，请重新输入")
    
    return board


def main():
    """主函数"""
    print("=" * 50)
    print("数独求解器 (Sudoku Solver)")
    print("=" * 50)
    
    # 选择输入方式
    print("\n请选择输入方式：")
    print("1. 使用示例数独")
    print("2. 手动输入数独")
    
    choice = input("\n请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        board = create_example_board()
        print("\n使用示例数独题目：")
    elif choice == "2":
        board = input_board()
    else:
        print("无效选择，使用示例数独")
        board = create_example_board()
    
    # 创建求解器
    solver = SudokuSolver(board)
    
    # 显示原始题目
    print("\n原始题目：")
    solver.print_board()
    
    # 解决数独
    print("正在求解...")
    if solver.solve():
        print("✓ 找到解决方案！")
        print("\n解决方案：")
        solver.print_board()
    else:
        print("✗ 无法解决此数独题目（可能题目无效）")


if __name__ == "__main__":
    main()


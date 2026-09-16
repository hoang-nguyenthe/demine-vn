"""Chia tập theo khối không gian.

Đây là chi tiết phương pháp quan trọng nhất của toàn bộ phần đánh giá, và cũng là
chi tiết dễ bị bỏ qua nhất.

Phân bố vật nổ còn sót có tương quan không gian rất mạnh: hai ô lưới cạnh nhau gần
như luôn cùng có hoặc cùng không có vật nổ, vì chúng nằm trong cùng một loạt bom.
Nếu chia tập huấn luyện và tập kiểm tra một cách ngẫu nhiên theo từng ô, thì mỗi ô
trong tập kiểm tra sẽ có hàng xóm nằm trong tập huấn luyện. Mô hình chỉ cần nội
suy từ hàng xóm là đã đạt chỉ tiêu rất cao, mà không hề học được quy luật nào.
Kết quả thu được sẽ đẹp và vô nghĩa.

Cách làm đúng là chia theo khối không gian liền mạch, đủ lớn để ranh giới giữa các
khối vượt quá tầm tương quan. Toàn bộ phần đánh giá của đề tài dùng cách chia này.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np


def make_spatial_blocks(
    col: np.ndarray, row: np.ndarray, n_blocks: int, grid_shape: Tuple[int, int]
) -> np.ndarray:
    """Gán mỗi ô lưới vào một khối không gian hình chữ nhật.

    Số khối theo mỗi chiều được chọn sao cho khối gần vuông nhất có thể, để không
    có chiều nào bị chia quá mảnh làm mất tác dụng của việc chia khối.
    """
    ny, nx = grid_shape
    side = max(1, int(round(np.sqrt(n_blocks))))
    n_bx = side
    n_by = max(1, int(np.ceil(n_blocks / side)))

    bx = np.minimum((np.asarray(col) * n_bx) // nx, n_bx - 1)
    by = np.minimum((np.asarray(row) * n_by) // ny, n_by - 1)
    return (by * n_bx + bx).astype(int)


def block_kfold(
    blocks: np.ndarray, n_folds: int, seed: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Sinh các lần chia huấn luyện và kiểm tra theo nhóm khối.

    Mỗi lần chia giữ lại toàn bộ một nhóm khối làm tập kiểm tra. Không có khối nào
    bị chia đôi giữa hai tập, nên không có rò rỉ thông tin qua ranh giới.
    """
    unique = np.unique(blocks)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    folds: List[np.ndarray] = np.array_split(order, max(1, n_folds))

    for held in folds:
        if held.size == 0:
            continue
        test_mask = np.isin(blocks, held)
        yield ~test_mask, test_mask


def spatial_holdout(
    blocks: np.ndarray, test_fraction: float = 0.3, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Một lần chia duy nhất theo khối, dùng cho các bước cần tập giữ lại cố định."""
    unique = np.unique(blocks)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    n_test = max(1, int(round(order.size * test_fraction)))
    held = order[:n_test]
    test_mask = np.isin(blocks, held)
    return ~test_mask, test_mask


def transfer_split(col: np.ndarray, grid_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Chia vùng nghiên cứu thành hai nửa đông và tây.

    Dùng cho phép thử chuyển vùng: huấn luyện trên một nửa, áp dụng cho nửa còn
    lại như thể đó là một địa bàn hoàn toàn mới chưa từng khảo sát.
    """
    _, nx = grid_shape
    west = np.asarray(col) < nx // 2
    return west, ~west

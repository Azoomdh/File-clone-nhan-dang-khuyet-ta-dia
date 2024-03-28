# %% [markdown]
# ### Multi-Label Classification

# %%
# import libraries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
"""
matplotlib là một thư viện toàn diện trong Python dùng để vẽ đồ thị1. Nó bao gồm nhiều module khác nhau1.

Trong khi đó, matplotlib.pyplot là một module con của matplotlib21. Nó cung cấp một giao diện trạng thái (state-machine interface) tới thư viện vẽ đồ thị cơ bản trong matplotlib1. Giao diện này giúp tạo ra các đồ thị phức tạp một cách dễ dàng hơn2.

from sklearn.preprocessing import StandardScaler
"""
Lệnh from sklearn.preprocessing import StandardScaler 
"""
trong Python có nghĩa là bạn đang nhập vào lớp StandardScaler từ module preprocessing của thư viện sklearn1.

StandardScaler là một công cụ tiền xử lý dữ liệu trong sklearn1. Nó chuẩn hóa các đặc trưng bằng cách loại bỏ giá trị trung bình và chia tỷ lệ để đạt đến phương sai đơn vị1. Điều này rất quan trọng đối với nhiều thuật toán học máy, vì chúng có thể hoạt động không tốt nếu các đặc trưng không tuân theo phân phối chuẩn (ví dụ: Gaussian với giá trị trung bình 0 và phương sai đơn vị)1.

Dưới đây là một ví dụ về cách sử dụng StandardScaler:

Python
"""

from sklearn.model_selection import train_test_split
"""
Lệnh from sklearn.model_selection import train_test_split trong Python có nghĩa là bạn đang nhập vào hàm train_test_split từ module model_selection của thư viện sklearn12.

Hàm train_test_split được sử dụng để chia dữ liệu của bạn thành hai tập con: tập huấn luyện và tập kiểm tra12. Điều này rất quan trọng trong quá trình xây dựng mô hình học máy, vì bạn muốn huấn luyện mô hình trên một tập dữ liệu và sau đó kiểm tra hiệu suất của nó trên một tập dữ liệu khác12.
"""

import tensorflow as tf
"""
TensorFlow là một thư viện mã nguồn mở được phát triển bởi Google Brain Team1. Nó được sử dụng rộng rãi trong học máy và học sâu để xây dựng và huấn luyện các mô hình1.

Một số tính năng chính của TensorFlow bao gồm:

Tính toán tensor: TensorFlow cho phép bạn thực hiện các phép tính trên tensor, đó là một cấu trúc dữ liệu giống như mảng nhiều chiều1.
Tính toán song song và phân tán: TensorFlow có thể tận dụng nhiều CPU hoặc GPU, và thậm chí có thể chạy trên nhiều máy tính1.
Xây dựng và huấn luyện mô hình học sâu: TensorFlow cung cấp các API để xây dựng và huấn luyện các mô hình học sâu phức tạp1.
Hỗ trợ cho các ngôn ngữ lập trình khác nhau: TensorFlow hỗ trợ nhiều ngôn ngữ lập trình, bao gồm Python, C++, và Java1.
"""

from tensorflow import keras
"""
Lệnh from tensorflow import keras trong Python được sử dụng để nhập mô-đun keras từ thư viện tensorflowkeras là một API cao cấp để xây dựng và huấn luyện các mô hình trong tensorflow
"""

from tensorflow.keras import Sequential 
"""
Lệnh from tensorflow.keras import Sequential trong Python được sử dụng để nhập mô hình Sequential từ thư viện tensorflow.keras.

Mô hình Sequential là một loại mô hình mạng nơ-ron trong Keras (một API của TensorFlow) cho phép bạn xây dựng mạng nơ-ron theo kiểu chồng chất, nghĩa là bạn có thể thêm các lớp một cách tuần tự. Điều này rất hữu ích khi bạn muốn xây dựng một mạng nơ-ron đơn giản mà không có các kết nối phức tạp giữa các lớp.

Dưới đây là một ví dụ về cách sử dụng mô hình Sequential
"""

from tensorflow.keras.layers import Dense,Dropout,Flatten
"""
Lệnh from tensorflow.keras.layers import Dense, Dropout, Flatten trong Python được sử dụng để nhập các lớp Dense, Dropout, và Flatten từ thư viện tensorflow.keras.layers.

Dense: Đây là lớp cơ bản nhất của mạng nơ-ron. Nó đại diện cho một lớp nơ-ron hoàn toàn kết nối, nghĩa là mỗi nơ-ron trong lớp này được kết nối với tất cả nơ-ron trong lớp trước và sau nó.
Dropout: Đây là một kỹ thuật chống quá khớp (overfitting) trong mạng nơ-ron. Quá khớp xảy ra khi mô hình học quá tốt trên dữ liệu huấn luyện nhưng không thể tổng quát hóa tốt trên dữ liệu kiểm tra. Lớp Dropout ngẫu nhiên “bỏ qua” một số nơ-ron trong quá trình huấn luyện, giúp mô hình không phụ thuộc quá mức vào bất kỳ nơ-ron cụ thể nào.
Flatten: Lớp này được sử dụng để làm phẳng đầu vào. Nó không có tham số để học; nó chỉ làm thay đổi hình dạng của dữ liệu. Điều này thường được sử dụng khi bạn cần chuyển đổi một tensor 3D thành một vector 1D.
"""

from tensorflow.keras.losses import BinaryCrossentropy
"""
Lệnh from tensorflow.keras.losses import BinaryCrossentropy trong Python được sử dụng để nhập hàm mất mát BinaryCrossentropy từ thư viện tensorflow.keras.losses.

BinaryCrossentropy là một hàm mất mát phổ biến được sử dụng trong các vấn đề phân loại nhị phân, nơi mục tiêu là dự đoán một trong hai lớp. Hàm này đo lường sự khác biệt giữa các dự đoán và nhãn thực tế.
"""

from tensorflow.keras.optimizers import Adam
"""
Lệnh from tensorflow.keras.optimizers import Adam trong Python được sử dụng để nhập lớp Adam từ thư viện tensorflow.keras.optimizers.

Adam là một thuật toán tối ưu hóa được sử dụng rộng rãi trong học sâu. Nó được thiết kế để kết hợp hai thuật toán tối ưu hóa khác là RMSProp và Stochastic Gradient Descent với momentum.

Thuật toán Adam hoạt động bằng cách cập nhật trọng số của mô hình dựa trên gradient của hàm mất mát, với sự điều chỉnh của tốc độ học và momentum. Điều này giúp thuật toán hội tụ nhanh hơn và hiệu quả hơn so với nhiều thuật toán tối ưu hóa khác.
"""


# %%
# read training data
df_train = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')
df_train.head()

"""
Lệnh này được sử dụng để đọc dữ liệu huấn luyện từ một tệp CSV và hiển thị 5 hàng đầu tiên của DataFrame.

Cụ thể, df_train = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv') sử dụng hàm read_csv từ thư viện pandas (pd) để đọc tệp CSV từ đường dẫn được chỉ định. Kết quả được lưu vào biến df_train dưới dạng một DataFrame.

df_train.head() sau đó được sử dụng để hiển thị 5 hàng đầu tiên của DataFrame. Điều này rất hữu ích để có cái nhìn tổng quan về dữ liệu mà bạn đang làm việc.

Ví dụ, bạn có thể xem các cột có trong DataFrame, kiểu dữ liệu của các cột, và một số giá trị mẫu từ mỗi cột. Điều này giúp bạn hiểu rõ hơn về cấu trúc và nội dung của dữ liệu của mình.
"""

# %% [markdown]
# ### ETL

# %%
print(df_train.info())

"""
Lệnh df_train.info() trong Python được sử dụng để in ra thông tin tổng quan về DataFrame df_train.

Thông tin này bao gồm:

Số lượng cột trong DataFrame.
Tên của mỗi cột.
Số lượng dữ liệu không null trong mỗi cột.
Kiểu dữ liệu của mỗi cột.
Dưới đây là một ví dụ về cách lệnh này hoạt động:
"""

# %%
df_train.isnull().sum()
"""
Lệnh df_train.isnull().sum() trong Python được sử dụng để kiểm tra số lượng giá trị null trong mỗi cột của DataFrame df_train.


"""

# %%
print(df_train.duplicated().sum())

"""
Lệnh df_train.duplicated().sum() trong Python được sử dụng để tính tổng số lượng các dòng bị trùng lặp trong DataFrame df_train.

Cụ thể, df_train.duplicated() sẽ trả về một Series Boolean, trong đó giá trị True tương ứng với các dòng bị trùng lặp và False tương ứng với các dòng không trùng lặp. Sau đó, .sum() sẽ tính tổng số lượng các giá trị True, tức là tổng số dòng bị trùng lặp.

Dưới đây là một ví dụ về cách lệnh này hoạt động:
"""

# %% [markdown]
# ### EDA

# %%
plt.figure(figsize=(18,14))
sns.heatmap(df_train.corr(),annot=True,fmt=".1f")

"""
Lệnh này được sử dụng để vẽ một biểu đồ heatmap (biểu đồ nhiệt) biểu diễn mối tương quan giữa các cột trong DataFrame df_train.

Cụ thể, các dòng lệnh có ý nghĩa như sau:

plt.figure(figsize=(18,14)): Đặt kích thước của hình vẽ là 18x14 (đơn vị là inch).
sns.heatmap(df_train.corr(),annot=True,fmt=".1f"): Vẽ một biểu đồ heatmap với seaborn (sns), sử dụng ma trận tương quan của DataFrame df_train (df_train.corr()). annot=True nghĩa là các giá trị tương quan sẽ được ghi trên biểu đồ. fmt=".1f" nghĩa là các giá trị này sẽ được làm tròn đến 1 chữ số sau dấu phẩy.
Ma trận tương quan là một ma trận vuông, trong đó mỗi cột và mỗi hàng tương ứng với một biến trong DataFrame. Giá trị tại hàng i, cột j là hệ số tương quan giữa biến thứ i và biến thứ j. Hệ số tương quan nằm trong khoảng từ -1 đến 1, trong đó -1 nghĩa là tương quan âm hoàn hảo, 1 nghĩa là tương quan dương hoàn hảo, và 0 nghĩa là không có tương quan.

Biểu đồ heatmap sẽ biểu diễn ma trận này dưới dạng một lưới các ô, với màu sắc của mỗi ô tương ứng với giá trị tương quan. Biểu đồ này rất hữu ích để nhìn thấy mối tương quan giữa các biến một cách trực quan.
"""

# %%
df_train.columns
"""
liệt kê các cột của biến df_train
"""

# %%
# split target column 
target = df_train[
    [
        'Pastry', 
        'Z_Scratch', 
        'K_Scatch', 
        'Stains',
        'Dirtiness', 
        'Bumps', 
        'Other_Faults'
    ]
]

"""
Lệnh này đang tạo một DataFrame mới có tên là target từ DataFrame df_train. DataFrame target này chỉ bao gồm các cột ‘Pastry’, ‘Z_Scratch’, ‘K_Scatch’, ‘Stains’, ‘Dirtiness’, ‘Bumps’, và ‘Other_Faults’ từ DataFrame df_train.

Đây thường là bước chuẩn bị dữ liệu khi chúng ta muốn tách nhãn (target) ra khỏi tập dữ liệu ban đầu để tiến hành các công việc như huấn luyện mô hình học máy. Trong trường hợp này, ‘Pastry’, ‘Z_Scratch’, ‘K_Scatch’, ‘Stains’, ‘Dirtiness’, ‘Bumps’, và ‘Other_Faults’ có thể là các nhãn mà chúng ta muốn dự đoán.
"""

# %%
# checking imbalances 
target.sum(axis=0)

"""
Lệnh target.sum(axis=0) được sử dụng để tính tổng của mỗi cột trong DataFrame target.

Trong trường hợp này, lệnh này được sử dụng với mục đích kiểm tra sự mất cân đối trong dữ liệu. Điều này thường được thực hiện trong các bài toán phân loại, khi chúng ta muốn xem xét sự phân bố của các lớp khác nhau trong tập dữ liệu.

Nếu một cột có tổng giá trị đáng kể lớn hơn hoặc nhỏ hơn so với các cột khác, điều đó có thể chỉ ra sự mất cân đối trong dữ liệu của lớp đó. Trong trường hợp đó, bạn có thể cần áp dụng các kỹ thuật tái cân đối dữ liệu như oversampling, undersampling, hoặc SMOTE.
"""

# %%
# split data features and target into variables
x = df_train.drop(['id','Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
       'Dirtiness', 'Bumps', 'Other_Faults'],axis=1)
y = target

"""
Lệnh này được sử dụng để chia dữ liệu thành hai phần: x (các đặc trưng) và y (mục tiêu).

x = df_train.drop(['id','Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'],axis=1): Lệnh này loại bỏ các cột ‘id’, ‘Pastry’, ‘Z_Scratch’, ‘K_Scatch’, ‘Stains’, ‘Dirtiness’, ‘Bumps’, và ‘Other_Faults’ khỏi DataFrame df_train và gán kết quả vào biến x. Điều này có nghĩa là x chứa tất cả các cột khác trong df_train ngoại trừ những cột đã được loại bỏ.
y = target: Lệnh này gán giá trị của biến target vào biến y. Biến target chưa được định nghĩa trong đoạn mã bạn đã cung cấp, nhưng thường target chứa giá trị mục tiêu mà chúng ta muốn dự đoán dựa trên các đặc trưng trong x.
Tóm lại, lệnh này đang chuẩn bị dữ liệu cho việc huấn luyện một mô hình học máy, với x là dữ liệu đầu vào (đặc trưng) và y là dữ liệu đầu ra (mục tiêu).
"""

# %%
# traning and testing splitting 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)

"""
Lệnh này được sử dụng để chia dữ liệu thành hai tập: tập huấn luyện và tập kiểm tra.

train_test_split(x, y, test_size=0.3,random_state=42): Lệnh này sử dụng hàm train_test_split từ thư viện sklearn.model_selection để chia dữ liệu x (đặc trưng) và y (mục tiêu) thành tập huấn luyện và tập kiểm tra.
test_size=0.3: Tham số này xác định tỷ lệ của dữ liệu được sử dụng cho tập kiểm tra. Trong trường hợp này, 30% dữ liệu sẽ được sử dụng cho tập kiểm tra và phần còn lại (70%) sẽ được sử dụng cho tập huấn luyện.
random_state=42: Tham số này đảm bảo rằng chia dữ liệu sẽ tạo ra cùng một kết quả mỗi khi chúng ta chạy lệnh này. Điều này giúp kết quả của quá trình huấn luyện và kiểm tra mô hình có thể tái tạo được.
X_train, X_test, y_train, y_test: Kết quả của hàm train_test_split được gán vào bốn biến này. X_train và y_train chứa dữ liệu huấn luyện (đặc trưng và mục tiêu tương ứng), trong khi X_test và y_test chứa dữ liệu kiểm tra.
Tóm lại, lệnh này đang chuẩn bị dữ liệu cho việc huấn luyện và kiểm tra một mô hình học máy.
"""

# %%
# scale training and test data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
Lệnh này được sử dụng để chuẩn hóa dữ liệu huấn luyện và kiểm tra.

sc = StandardScaler(): Lệnh này tạo một đối tượng StandardScaler từ thư viện sklearn.preprocessing và gán nó vào biến sc. StandardScaler được sử dụng để chuẩn hóa dữ liệu bằng cách loại bỏ giá trị trung bình và chia tỷ lệ dữ liệu về độ lệch chuẩn.
X_train = sc.fit_transform(X_train): Lệnh này sử dụng phương thức fit_transform của đối tượng StandardScaler để tính toán giá trị trung bình và độ lệch chuẩn của dữ liệu huấn luyện (X_train), sau đó chuẩn hóa dữ liệu này. Kết quả sau khi chuẩn hóa được gán trở lại vào X_train.
X_test = sc.transform(X_test): Lệnh này sử dụng phương thức transform của đối tượng StandardScaler để chuẩn hóa dữ liệu kiểm tra (X_test) dựa trên giá trị trung bình và độ lệch chuẩn đã được tính từ dữ liệu huấn luyện. Kết quả sau khi chuẩn hóa được gán vào X_test.
Tóm lại, lệnh này đang chuẩn hóa dữ liệu huấn luyện và kiểm tra để chuẩn bị cho việc huấn luyện và kiểm tra mô hình học máy. Chuẩn hóa dữ liệu là một bước quan trọng trong quá trình tiền xử lý dữ liệu, giúp cải thiện hiệu suất và ổn định của mô hình học máy.
"""

# %%
X_train.shape,X_test.shape,y_train.shape,y_test.shape
"""
Lệnh này được sử dụng để lấy kích thước (số hàng và số cột) của các tập dữ liệu X_train, X_test, y_train, và y_test.

X_train.shape: Trả về một tuple biểu diễn kích thước của X_train (số lượng mẫu huấn luyện và số lượng đặc trưng).
X_test.shape: Trả về một tuple biểu diễn kích thước của X_test (số lượng mẫu kiểm tra và số lượng đặc trưng).
y_train.shape: Trả về một tuple biểu diễn kích thước của y_train (số lượng mẫu huấn luyện).
y_test.shape: Trả về một tuple biểu diễn kích thước của y_test (số lượng mẫu kiểm tra).
Tóm lại, lệnh này giúp bạn kiểm tra kích thước của các tập dữ liệu sau khi đã chia thành tập huấn luyện và tập kiểm tra. Điều này hữu ích để đảm bảo rằng dữ liệu đã được chia đúng cách và sẵn sàng cho quá trình huấn luyện mô hình học máy.
"""

# %%
# finding class weights 
class_weights = {i: len(y_train.values) / (np.sum(y_train.values[:, i]) + 1) for i in range(y_train.shape[1])}
class_weights

"""
Lệnh này được sử dụng để tính toán trọng số của các lớp trong tập dữ liệu huấn luyện y_train.

class_weights = {i: len(y_train.values) / (np.sum(y_train.values[:, i]) + 1) for i in range(y_train.shape[1])}: Lệnh này tạo một từ điển class_weights trong đó khóa là chỉ số của lớp và giá trị là trọng số của lớp đó. Trọng số của mỗi lớp được tính bằng cách lấy tổng số mẫu huấn luyện chia cho (số lượng mẫu thuộc lớp đó cộng 1). Cộng 1 vào mẫu để tránh chia cho 0 trong trường hợp không có mẫu nào thuộc lớp đó.
class_weights: Lệnh này in ra từ điển class_weights đã được tính toán.
Tóm lại, lệnh này đang tính toán trọng số của các lớp trong tập dữ liệu huấn luyện. Trọng số lớp có thể được sử dụng trong quá trình huấn luyện mô hình học máy để cân bằng dữ liệu, đặc biệt khi có sự mất cân đối giữa các lớp. Trọng số lớp cao hơn sẽ cho mô hình biết rằng nó nên tập trung hơn vào các mẫu thuộc lớp đó.
"""

# %%
# Ann Model building
model = Sequential([
    Flatten(),
    Dense(256, activation='relu', input_shape=(27,)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss=BinaryCrossentropy(),
    metrics=["accuracy"]
)

"""
Lệnh này được sử dụng để xây dựng và biên dịch một mô hình mạng nơ-ron nhân tạo (Artificial Neural Network - ANN) sử dụng thư viện Keras.

model = Sequential([...]): Lệnh này tạo một mô hình tuần tự (Sequential) mới, là một loại mô hình Keras đơn giản nhất gồm một ngăn xếp tuyến tính các lớp.
Flatten(): Lớp này chuyển đổi một tensor đầu vào thành một tensor 1D (làm phẳng dữ liệu).
Dense(256, activation='relu', input_shape=(27,)): Lớp này tạo một lớp dày đặc (Dense) với 256 nơ-ron và hàm kích hoạt ReLU. Tham số input_shape=(27,) chỉ ra rằng dữ liệu đầu vào cho mô hình sẽ có 27 đặc trưng.
Dropout(0.2): Lớp này tạo một lớp Dropout, trong đó 20% (0.2) các nơ-ron sẽ bị “tắt” trong quá trình huấn luyện, giúp ngăn chặn hiện tượng quá khớp (overfitting).
Dense(128, activation='relu'), Dense(64, activation='relu'), Dense(7, activation='sigmoid'): Các lớp này tạo thêm các lớp dày đặc với số lượng nơ-ron tương ứng là 128, 64 và 7. Hàm kích hoạt của các lớp cuối cùng là ‘sigmoid’, thích hợp cho bài toán phân loại nhiều lớp.
model.compile(...): Lệnh này biên dịch mô hình, trong đó:
optimizer=Adam(learning_rate=0.01): Sử dụng thuật toán tối ưu Adam với tốc độ học là 0.01.
loss=BinaryCrossentropy(): Sử dụng hàm mất mát Binary Crossentropy, thích hợp cho bài toán phân loại nhị phân.
metrics=["accuracy"]: Đánh giá mô hình dựa trên độ chính xác (accuracy) trong quá trình huấn luyện và kiểm tra.
Tóm lại, lệnh này đang xây dựng và biên dịch một mô hình mạng nơ-ron nhân tạo để chuẩn bị cho quá trình huấn luyện. Mô hình này bao gồm nhiều lớp dày đặc và lớp dropout để ngăn chặn hiện tượng quá khớp. Mô hình sau cùng sẽ có 7 nơ-ron ở lớp cuối cùng, với hàm kích hoạt ‘sigmoid’, phù hợp cho bài toán phân loại nhiều lớp. Mô hình sẽ được tối ưu hóa bằng thuật toán Adam và đánh giá dựa trên độ chính xác.
"""

# %%
# traning started 
history = model.fit(
    X_train,y_train,
    epochs=50,
    verbose=1,
    batch_size=28,
    validation_data=(X_test,y_test),
    class_weight=class_weights
)

"""
Lệnh này được sử dụng để bắt đầu quá trình huấn luyện mô hình mạng nơ-ron nhân tạo (Artificial Neural Network - ANN) mà bạn đã xây dựng.

history = model.fit(...): Lệnh này sử dụng phương thức fit của đối tượng mô hình để huấn luyện mô hình. Kết quả của quá trình huấn luyện (bao gồm thông tin về độ mất mát và độ chính xác sau mỗi epoch) được lưu vào biến history.
X_train, y_train: Đây là dữ liệu huấn luyện, bao gồm đặc trưng (X_train) và mục tiêu (y_train).
epochs=50: Đây là số lượng lần mô hình sẽ đi qua toàn bộ tập dữ liệu huấn luyện. Mỗi lần như vậy được gọi là một epoch.
verbose=1: Đây là mức độ thông báo trong quá trình huấn luyện. Giá trị 1 có nghĩa là in ra thông tin sau mỗi epoch.
batch_size=28: Đây là số lượng mẫu sẽ được truyền qua mô hình trước khi mô hình cập nhật trọng số.
validation_data=(X_test, y_test): Đây là dữ liệu sẽ được sử dụng để đánh giá mô hình sau mỗi epoch.
class_weight=class_weights: Đây là trọng số của các lớp trong dữ liệu huấn luyện. Trọng số này sẽ được sử dụng trong quá trình huấn luyện để cân bằng dữ liệu.
Tóm lại, lệnh này bắt đầu quá trình huấn luyện mô hình ANN với dữ liệu huấn luyện, số lượng epoch, kích thước batch, dữ liệu kiểm tra và trọng số lớp đã được chỉ định. Quá trình huấn luyện này sẽ cố gắng tối ưu hóa mô hình để giảm thiểu độ mất mát trên dữ liệu huấn luyện và tăng độ chính xác trên dữ liệu kiểm tra.
"""

# %%
# visualization of loss and auc score
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

"""
Lệnh này được sử dụng để vẽ hai biểu đồ: một cho độ chính xác của mô hình (accuracy) và một cho độ mất mát của mô hình (loss), cả hai đều theo số lượng epoch. Biểu đồ này giúp bạn theo dõi quá trình học của mô hình qua từng epoch.

plt.figure(figsize=(12, 4)): Tạo một hình mới với kích thước 12x4 (đơn vị là inch).
plt.subplot(1, 2, 1): Chia hình thành một lưới 1x2 và tạo một subplot ở vị trí đầu tiên.
plt.plot(history.history['accuracy']) và plt.plot(history.history['val_accuracy']): Vẽ đường biểu diễn độ chính xác của mô hình trên tập huấn luyện và tập kiểm tra qua từng epoch.
plt.title('Model accuracy'), plt.ylabel('accuracy'), plt.xlabel('Epoch'), và plt.legend(['Train', 'Validation'], loc='upper left'): Đặt tiêu đề, nhãn cho trục y, trục x và chú thích cho biểu đồ độ chính xác.
plt.subplot(1, 2, 2): Tạo một subplot ở vị trí thứ hai trong lưới 1x2.
plt.plot(history.history['loss']) và plt.plot(history.history['val_loss']): Vẽ đường biểu diễn độ mất mát của mô hình trên tập huấn luyện và tập kiểm tra qua từng epoch.
plt.title('Model Loss'), plt.ylabel('Loss'), plt.xlabel('Epoch'), và plt.legend(['Train', 'Validation'], loc='upper left'): Đặt tiêu đề, nhãn cho trục y, trục x và chú thích cho biểu đồ độ mất mát.
plt.tight_layout(): Điều chỉnh vị trí của các subplot để chúng không bị chồng lên nhau.
plt.show(): Hiển thị hình với hai biểu đồ đã vẽ.
Tóm lại, lệnh này giúp bạn vẽ và hiển thị hai biểu đồ để theo dõi quá trình học của mô hình qua từng epoch, một biểu đồ cho độ chính xác và một biểu đồ cho độ mất mát, cả hai đều so sánh giữa tập huấn luyện và tập kiểm tra. Biểu đồ này rất hữu ích để kiểm tra xem mô hình có đang bị quá khớp (overfitting) hay không.
"""

# %%
# reading test data
df_test = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv')
df_test.head()

"""
Lệnh này được sử dụng để đọc dữ liệu kiểm tra từ một tệp CSV và hiển thị 5 hàng đầu tiên của DataFrame.

df_test = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv'): Lệnh này sử dụng hàm read_csv từ thư viện pandas để đọc tệp CSV từ đường dẫn đã cho và lưu dữ liệu vào DataFrame df_test.
df_test.head(): Lệnh này sử dụng phương thức head của DataFrame để hiển thị 5 hàng đầu tiên của df_test.
Tóm lại, lệnh này đang đọc dữ liệu kiểm tra từ tệp CSV và hiển thị 5 hàng đầu tiên của dữ liệu. Điều này giúp bạn nhanh chóng kiểm tra và hiểu cấu trúc cơ bản của dữ liệu kiểm tra.
"""

# %%
# store id in a variable
id_test=df_test.id
id_test.values

"""
Lệnh này được sử dụng để lưu trữ giá trị của cột ‘id’ từ DataFrame df_test vào một biến id_test và sau đó in ra các giá trị của id_test.

id_test=df_test.id: Lệnh này lấy cột ‘id’ từ DataFrame df_test và lưu trữ nó vào biến id_test.
id_test.values: Lệnh này trả về một mảng Numpy chứa tất cả các giá trị trong id_test.
Tóm lại, lệnh này đang lưu trữ giá trị ‘id’ từ dữ liệu kiểm tra vào một biến và sau đó in ra các giá trị đó. Điều này có thể hữu ích khi bạn muốn sử dụng các giá trị ‘id’ này sau này, ví dụ như khi bạn muốn kết hợp dự đoán của mô hình với các ‘id’ tương ứng để tạo ra tệp kết quả cuối cùng.
"""

# %%
# drop id column
df_test.drop('id',axis=1,inplace=True)

"""
Lệnh này được sử dụng để loại bỏ cột ‘id’ khỏi DataFrame df_test.

df_test.drop('id',axis=1,inplace=True): Lệnh này sử dụng phương thức drop của DataFrame để loại bỏ cột ‘id’. Tham số axis=1 chỉ ra rằng chúng ta muốn loại bỏ một cột (không phải hàng). Tham số inplace=True chỉ ra rằng chúng ta muốn thay đổi trực tiếp DataFrame df_test thay vì tạo một DataFrame mới.
Tóm lại, lệnh này đang loại bỏ cột ‘id’ khỏi dữ liệu kiểm tra. Điều này có thể hữu ích khi bạn không muốn sử dụng ‘id’ như một đặc trưng trong quá trình dự đoán của mô hình học máy.
"""

# %%
# scale test data
test = sc.transform(df_test)

"""
Lệnh này được sử dụng để chuẩn hóa dữ liệu kiểm tra (df_test) sử dụng đối tượng StandardScaler (sc) đã được khớp với dữ liệu huấn luyện trước đó.

test = sc.transform(df_test): Lệnh này sử dụng phương thức transform của đối tượng StandardScaler để chuẩn hóa dữ liệu kiểm tra (df_test) dựa trên giá trị trung bình và độ lệch chuẩn đã được tính từ dữ liệu huấn luyện. Kết quả sau khi chuẩn hóa được gán vào biến test.
Tóm lại, lệnh này đang chuẩn hóa dữ liệu kiểm tra dựa trên tham số đã được học từ dữ liệu huấn luyện. Điều này giúp đảm bảo rằng dữ liệu kiểm tra được xử lý tương tự như dữ liệu huấn luyện, điều này là cần thiết để đưa ra dự đoán chính xác từ mô hình đã được huấn luyện.
"""

# %%
# prediction on test data
y_pred = model.predict(test)
y_pred

"""
Lệnh này được sử dụng để dự đoán kết quả trên dữ liệu kiểm tra (test) sử dụng mô hình mạng nơ-ron nhân tạo (Artificial Neural Network - ANN) mà bạn đã huấn luyện.

y_pred = model.predict(test): Lệnh này sử dụng phương thức predict của đối tượng mô hình để dự đoán kết quả trên dữ liệu kiểm tra (test). Kết quả dự đoán được lưu vào biến y_pred.
y_pred: Lệnh này in ra các giá trị dự đoán đã được lưu trong y_pred.
Tóm lại, lệnh này đang dự đoán kết quả trên dữ liệu kiểm tra sử dụng mô hình ANN đã được huấn luyện và sau đó in ra các giá trị dự đoán đó. Điều này hữu ích khi bạn muốn xem kết quả dự đoán của mô hình trên dữ liệu chưa được nhìn thấy trước đó
"""

# %%
# round it two decimal 
y_pred = np.round(y_pred,2)
"""
làm tròn kết quả của y_pred trong 2 chữ số thập phân, tránh hiển thị quá nhiều số phía sau dấu phẩy
"""

# %%
len(y_pred),len(id_test.values),y_pred.shape

"""
Lệnh này được sử dụng để lấy độ dài của y_pred và id_test.values, cũng như kích thước của y_pred.

len(y_pred): Lệnh này trả về số lượng phần tử trong y_pred, tức là số lượng dự đoán mà mô hình đã tạo ra cho dữ liệu kiểm tra.
len(id_test.values): Lệnh này trả về số lượng phần tử trong id_test.values, tức là số lượng ‘id’ trong dữ liệu kiểm tra.
y_pred.shape: Lệnh này trả về một tuple biểu diễn kích thước của y_pred (số lượng mẫu kiểm tra và số lượng lớp dự đoán).
Tóm lại, lệnh này giúp bạn kiểm tra số lượng dự đoán mà mô hình đã tạo ra, so sánh nó với số lượng ‘id’ trong dữ liệu kiểm tra, và xem kích thước của y_pred để hiểu rõ hơn về cấu trúc của dự đoán. Điều này hữu ích khi bạn muốn đảm bảo rằng mô hình đã tạo ra đúng số lượng dự đoán và cấu trúc dự đoán phù hợp với yêu cầu.
"""

# %% [markdown]
# ### Submission 

# %%
# checking submission file look like
sample = pd.read_csv('/kaggle/input/playground-series-s4e3/sample_submission.csv')
sample.head()

"""
Lệnh này được sử dụng để đọc tệp CSV mẫu cho việc nộp kết quả và hiển thị 5 hàng đầu tiên của DataFrame.

sample = pd.read_csv('/kaggle/input/playground-series-s4e3/sample_submission.csv'): Lệnh này sử dụng hàm read_csv từ thư viện pandas để đọc tệp CSV từ đường dẫn đã cho và lưu dữ liệu vào DataFrame sample.
sample.head(): Lệnh này sử dụng phương thức head của DataFrame để hiển thị 5 hàng đầu tiên của sample.
Tóm lại, lệnh này đang đọc tệp CSV mẫu cho việc nộp kết quả và hiển thị 5 hàng đầu tiên của dữ liệu. Điều này giúp bạn nhanh chóng kiểm tra và hiểu cấu trúc cơ bản của tệp nộp kết quả mẫu. Điều này rất hữu ích khi bạn cần chuẩn bị tệp nộp kết quả cuối cùng cho cuộc thi hoặc dự án.
"""

# %%
# submission data frame created
sub = pd.DataFrame(
    {
        'id' : id_test.values,
        'Pastry':y_pred[:,0],
        'Z_Scratch':y_pred[:,1],
        'K_Scatch':y_pred[:,2],
        'Stains':y_pred[:,3],
        'Dirtiness':y_pred[:,4],
        'Bumps':y_pred[:,5],
        'Other_Faults':y_pred[:,6]
    }
)
sub.head()

"""
Lệnh này được sử dụng để tạo một DataFrame mới cho việc nộp kết quả dựa trên các dự đoán y_pred và id_test.values.

sub = pd.DataFrame({...}): Lệnh này tạo một DataFrame mới và gán nó vào biến sub. DataFrame này bao gồm các cột tương ứng với ‘id’ và các lớp mục tiêu (‘Pastry’, ‘Z_Scratch’, ‘K_Scatch’, ‘Stains’, ‘Dirtiness’, ‘Bumps’, ‘Other_Faults’).
'id' : id_test.values: Đây là cột ‘id’ của DataFrame, chứa các giá trị ‘id’ từ dữ liệu kiểm tra.
'Pastry':y_pred[:,0], 'Z_Scratch':y_pred[:,1], ...: Đây là các cột mục tiêu của DataFrame, chứa các dự đoán tương ứng của mô hình cho mỗi lớp mục tiêu.
sub.head(): Lệnh này sử dụng phương thức head của DataFrame để hiển thị 5 hàng đầu tiên của sub.
Tóm lại, lệnh này đang tạo một DataFrame mới cho việc nộp kết quả dựa trên các dự đoán của mô hình và các ‘id’ tương ứng từ dữ liệu kiểm tra, sau đó hiển thị 5 hàng đầu tiên của DataFrame này. 
"""

# %%
# saved in csv file
sub.to_csv('submission.csv',index=False)
Lệnh này được sử dụng để lưu DataFrame sub vào một tệp CSV.

sub.to_csv('submission.csv',index=False): Lệnh này sử dụng phương thức to_csv của DataFrame để lưu sub vào một tệp CSV có tên là ‘submission.csv’. Tham số index=False chỉ ra rằng chúng ta không muốn lưu các chỉ số của DataFrame vào tệp CSV.
Tóm lại, lệnh này đang lưu DataFrame sub vào một tệp CSV để bạn có thể nộp kết quả dự đoán của mô hình. Tệp CSV này có thể được tải lên một nền tảng như Kaggle để tham gia cuộc thi hoặc để chia sẻ kết quả với người khác. Điều này rất hữu ích khi bạn cần lưu và chia sẻ kết quả dự đoán của mô hình học máy.



import matplotlib.pyplot as plt


def visualization(test_y, pred_y):
    # 실제값, 예측값 비교 시각화

    plt.figure(figsize=(30, 10))
    plt.plot(test_y, color='red', label='real target y')
    plt.plot(pred_y, color='blue', label='predict y')
    plt.legend()
    plt.show()
    
    # 파일로 저장

    plt.savefig('lstm 모델 시각화.png')
# 밑바닥부터 시작하는 딥러닝 3
### 매일 최소 1 Step을 커밋합니다.
#### 4단계 - 수치 미분
- 컴퓨터는 극한을 취급할 수 없으니 매우 작은 값을 이용하여 함수의 변화량을 구하는 방법을 `numerical differentiation`이라고 함
- 근사 오차를 줄이는 방법으로 `centered difference`을 사용
- `numerical differentiation`은 계산량이 많은 단점이 있음
- 신경망에서는 매개변수가 수백만 개 이상이기 때문에 이 모두를 `numerical differentiation`로 구하는 것은 현실적이지 않음
- 따라서 역전파를 사용하며 `numerical differentiation`는 `gradient checking`에 사용함
#### 5단계 - 역전파 이론
- `chain rule`에 따르면 합성 함수의 미분은 구성 함수 각각을 미분한 후 곱한 것과 같음
- `loss function`의 각 매개변수에 대한 미분을 계산해야 함
- 이 경우 미분값을 출력에서 입력 방향으로 전파하면 한 번의 전파만으로 모든 매개변수에 대한 미분 계산 가능
#### 6단계 - 수동 역전파
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
- 역전파 시에는 순전파 시 이용한 데이터가 필요
- 역전파를 구하려면 먼저 순전파를 하고 이때 각 함수가 입력 변수의 값을 기억해둬야 함
#### 6단계 - 수동 역전파
#### 7단계 - 역전파 자동화
- `Define-by-Run`이란 딥러닝에서 수행하는 계산들을 계산 시점에 연결하는 방식
- 실제 계산이 이루어질 때 변수에 관련 연결을 기록하는 방식
#### 8단계 - 재귀에서 반복문으로
- 재귀는 함수를 재귀적으로 호출할 때마다 중간 결과를 메모리에 유지하면서 처리
#### 9단계 - 함수를 더 편리하게
- `ndarray`만 취급하기 위해 `scalar` 값이 입력되거나 출력되면 적절히 변환
#### 10단계 - 파이썬 단위 테스트
- 테스트를 해야 실수를 예방할 수 있으며 테스트를 자동화해야 소프트웨어의 품질을 유지할 수 있음
- `discover` 명령어로 테스트 파일들을 한꺼번에 실행할 수 있음
#### 자동 미분
- 미분은 `수치 미분`, `기호 미분`, `자동 미분` 3가지로 나눌 수 있음
- `자동 미분`은 forward 모드와 reverse 모드로 나뉘어짐
#### 11단계 - 가변 길이 인수(순전파 편)
- 함수에 따라 입력이 여러 개인 경우도 있고 출력이 여러 개인 경우도 있음
- 가변 길이 입출력을 표현하기 위해 변수들을 리스트에 넣어 처리
- 리스트를 생성할 때 `list comprehension`을 사용
#### 12단계 - 가변 길이 인수(개선 편)
- 함수를 사용하기 쉽도록 개선
- 함수를 구현하기 쉽도록 개선
#### 13단계 - 가변 길이 인수(역전파 편)
- 가변 길이 인수에 대응한 역전파
#### 14단계 - 같은 변수 반복 사용
- 같은 변수를 반복해서 사용할 경우 의도대로 동작하지 않을 수 있음
#### 15단계 - 복잡한 계산 그래프(이론 편)
- 현재 구조에서는 이런 복잡한 연결의 역전파를 제대로 할 수 없음
- `위성 정렬` 알고리즘을 사용해 노드의 연결 방법을 기초로 정렬할 수 있음
#### 16단계 - 복잡한 계산 그래프(구현 편)
- 몇 번째 '세대'의 함수인지 나타내는 변수를 추가
#### 17단계 - 메모리 관리와 순환 참조
- 파이썬은 참조 수를 세는 방법과 세대를 기준으로 쓸모 없어진 객체를 회수하는 방이 있음
- 모든 객체는 참조 카운트가 0인 상태로 생성, 다른 객체가 참조할 때마다 1씩 증가
- 객체에 대한 참조가 끊길 때마다 1만큼 감소하다가 0이 되면 파이썬 인터프리터가 회수
- 대입 연산자를 사용, 함수에 인수로 전달, 컨테이너 타입 객체에 추가할 때 증가
- `약한 참조`란 다른 객체를 참조하되 참조 카운트는 증가시키지 않는 기능
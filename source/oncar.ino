#define LEFT_AHEAD 10
#define LEFT_BACK 9
#define RIGHT_AHEAD 12
#define RIGHT_BACK 13
#define STEER 14

int duty = 1230;
int CYC = 5;
int STOP = 0;     //设置不同的速度，用于转不同的角度
int TURN1 = 50;
int TURN2 = 150;
int TURN3 = 130;
int RUN0 = 100;
int RUN00 = 120;
int RUN1 = 110;
int RUN2 = 120;
int RUN3 = 170;

void turnLeft1();
void turnLeft2();
void turnLeft3();
void turnLeft4();
void turnRight1();
void turnRight2();
void turnRight3();
void turnRight4();
void goAhead();
void park();
void goBack();

void setup(){
Serial.begin(9600);
pinMode(LEFT_AHEAD,OUTPUT);
pinMode(LEFT_BACK, OUTPUT);
pinMode(RIGHT_AHEAD, OUTPUT);
pinMode(RIGHT_BACK, OUTPUT);
digitalWrite(LEFT_AHEAD, LOW);
digitalWrite(LEFT_BACK, LOW);
digitalWrite(RIGHT_AHEAD, LOW);
digitalWrite(RIGHT_BACK, LOW);;
pinMode(STEER, OUTPUT);
digitalWrite(STEER, LOW);
for (int i=0; i<100; i++) {
digitalWrite(STEER,LOW);
delayMicroseconds(duty);
digitalWrite(STEER,HIGH);
delayMicroseconds(10000-duty);
delayMicroseconds(6550);
}
}

char operation = ' ';
void loop()
{                           
if (Serial.available() > 0) { 
operation = Serial.read();
 switch(operation)
    {
      case 'M':
        turnLeft();
        break;
      case 'S':
        turnRight();
        break;
      case 'A':
        goAhead();
        break;
      case 'B':
        goBack();
        break;
      case 'P':
        park();
        break;

}
}
}

void steer(float x){
int tmp = x*duty;
for (int i=0; i<CYC; i++) {
digitalWrite(STEER,LOW);
delayMicroseconds(tmp);
digitalWrite(STEER,HIGH);
delayMicroseconds(10000-tmp);
delayMicroseconds(6550);
}

}
void goAhead(){            //直走，50
digitalWrite(LEFT_BACK, LOW);
digitalWrite(RIGHT_BACK, LOW);
analogWrite(LEFT_AHEAD,RUN0);
analogWrite(RIGHT_AHEAD,RUN0);
steer(1.0);
}

void turnLeft(){              //左转2，大角度
digitalWrite(LEFT_AHEAD,LOW);
digitalWrite(RIGHT_BACK,LOW);
analogWrite(RIGHT_AHEAD,RUN1);
analogWrite(LEFT_BACK,TURN1);
steer(1.1);
}
void turnRight(){             //右转2，大角度
digitalWrite(LEFT_BACK,LOW);
digitalWrite(RIGHT_AHEAD,LOW);
analogWrite(LEFT_AHEAD,RUN1);
analogWrite(RIGHT_BACK,TURN1);
steer(1.1);
}
void park(){                   //停下
digitalWrite(LEFT_BACK, LOW);
digitalWrite(RIGHT_BACK, LOW);
digitalWrite(LEFT_AHEAD, LOW);
digitalWrite(RIGHT_AHEAD, LOW);
steer(1);
}

void goBack(){                //后退
digitalWrite(LEFT_AHEAD,LOW);
digitalWrite(RIGHT_AHEAD,LOW);
analogWrite(LEFT_BACK,RUN0);
analogWrite(RIGHT_BACK,RUN0);
steer(1);
}

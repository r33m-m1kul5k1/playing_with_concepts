#include<stdio.h>
void hanoi(int nrings, int from, int to, int via)
{
	if (1 < nrings) {
		hanoi(nrings - 1, from, via, to);
		printf("Move a ring from tower %d to tower %d.\n", from, to);
		hanoi(nrings - 1, via, to, from);
	} else {
		printf("Move a ring from tower %d to tower %d.\n", from, to);
	}
	
	return;
}

int main(int argc, char *argv[]) {
	hanoi(atoi(argv[1]), 1, 2, 3);
	return 0;
}

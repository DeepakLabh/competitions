	#include<stdio.h>

	/* Our structure */
	struct rec
	{
		int x,y,z;
	};

	int main()
	{
		int counter;
		FILE *ptr_myfile;
		struct rec my_record;

		ptr_myfile=fopen("/home/deepak/deepak/competitions/kaggle/quora/word2vec/data/text8-vector.bin","rb");
		if (!ptr_myfile)
		{
			printf("Unable to open file!");
			return 1;
		}
		for ( counter=1; counter <= 100; counter++)
		{
			fread(&my_record,sizeof(struct rec),1,ptr_myfile);
			printf("%d\t",my_record.x);
		}
		fclose(ptr_myfile);
		return 0;
	}

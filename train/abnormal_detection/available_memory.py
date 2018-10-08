from utils.read_csv import Reader

if __name__ == "__main__":
    # test class
    reader = Reader("../../datasets/db_os_stat(180123-180128).csv", "FREE_MEM_SIZE")
    print(reader.data)




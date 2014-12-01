d_human_mask <= set the all one

template<T>
class M_Queue {
	int head, tail;
	int count;
	int size;
public:
	T *queue;
	M_Queue(int size) {
		head = tail = 0;
		count = 0;
		queue = new T[size];
	}

	T* enqueue() { 
		if (count <= size) {
			T* result = queue + tail;
			tail_inc();	
			return result;
		}
		return NULL;
	}
	T* tail()  { return count <= size? queue + tail : NULL; }
	T* front() { return count > 0 ? queue + head : NULL; }
	T* front_2() { return count > 1 ? queue + (head + 1) % size : NULL; }
	void tail_inc() 
	{
		tail = (tail + 1) % size;
		count++;
	}

	void tail_dec()
	{
		tail = (tail - 1) % size;
		count++;
	}

	void head_dec() 
	{
		head = (head + 1) % size;
		count--;
	}

	T* dequeue() { 
		if (count > 0) {
			T* result = queue + head;
			head_dec();	
			return result;
		}
		return NULL;
	}
};

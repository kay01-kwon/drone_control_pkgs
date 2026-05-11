#ifndef CircularBuffer_HPP
#define CircularBuffer_HPP

#include <iostream>
#include <vector>
#include <optional>
#include <mutex>
#include <thread>

template <typename T>
class CircularBuffer{

    public:

    CircularBuffer();

    CircularBuffer(size_t capacity);

    bool reserve(size_t new_capacity);

    void push_back(const T& item);

    void push_back(T&& item);

    std::optional<T> pop();

    std::optional<T> at(size_t index) const;

    std::optional<T>& operator[](size_t index);

    std::optional<T> get_latest() const;

    std::optional<T> get_oldest() const;

    void clear();

    size_t size() const;

    size_t capacity() const;

    bool is_empty() const;

    bool is_full() const;

    private:

    bool empty_unsafe() const;

    size_t capacity_;
    std::vector<T> buffer_;
    size_t head_;
    size_t tail_;
    size_t size_;
    bool is_full_;
    mutable std::mutex mutex_;
};


#endif // CircularBuffer_HPP

template <typename T>
CircularBuffer<T>::CircularBuffer()
: capacity_(10),
buffer_(capacity_),
head_(0),
tail_(0),
size_(0),
is_full_(false) 
{

}

template <typename T>
CircularBuffer<T>::CircularBuffer(size_t capacity)
: capacity_(capacity),
buffer_(capacity),
head_(0),
tail_(0),
size_(0),
is_full_(false) 
{

}

template <typename T>
bool CircularBuffer<T>::reserve(size_t new_capacity){
    std::lock_guard<std::mutex> lock(mutex_);
    
    if(new_capacity < size_){
        return false;
    }
    
    // Create new buffer
    std::vector<T> new_buffer(new_capacity);
    
    // Copy existing elements to new buffer
    for(size_t i = 0; i < size_; ++i){
        size_t old_index = (tail_ + i) % capacity_;
        new_buffer[i] = std::move(buffer_[old_index]);
    }
    
    // Update buffer and indices
    buffer_ = std::move(new_buffer);
    capacity_ = new_capacity;
    tail_ = 0;
    head_ = size_;
    is_full_ = (size_ == capacity_);
    
    return true;
}

template <typename T>
void CircularBuffer<T>::push_back(const T& item)
{
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_[head_] = item;

    if(is_full_){
        tail_ = (tail_ + 1) % capacity_;
    }

    head_ = (head_ + 1) % capacity_;

    if(!is_full_)
        ++size_;

    is_full_ = (head_ == tail_);
}

template <typename T>
void CircularBuffer<T>::push_back(T&& item){
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_[head_] = std::move(item);
    
    if(is_full_){
        tail_ = (tail_ + 1) % capacity_;
    }

    head_ = (head_ + 1) % capacity_;

    if(!is_full_)
        ++size_;
    is_full_ = (head_ == tail_);
}

template <typename T>
std::optional<T> CircularBuffer<T>::pop(){
    std::lock_guard<std::mutex> lock(mutex_);
    
    if(empty_unsafe()){
        return std::nullopt;
    }

    T item = std::move(buffer_[tail_]);
    is_full_ = false;
    tail_ = (tail_ + 1) % capacity_;
    --size_;
    return item;
}

template <typename T>
std::optional<T> CircularBuffer<T>::at(size_t index) const{
    std::lock_guard<std::mutex> lock(mutex_);
    if(index >= size_){
        return std::nullopt;
    }
    size_t real_index = (tail_ + index) % capacity_;
    return buffer_[real_index];
}

template <typename T>
std::optional<T>& CircularBuffer<T>::operator[](size_t index){

    return at(index);
}

template <typename T>
std::optional<T> CircularBuffer<T>::get_latest() const{
    std::lock_guard<std::mutex> lock(mutex_);
    if(empty_unsafe()){
        return std::nullopt;
    }

    size_t latest_index = (head_ == 0) ? capacity_ - 1 : head_ - 1;
    return buffer_[latest_index];
}

template <typename T>
std::optional<T> CircularBuffer<T>::get_oldest() const{
    std::lock_guard<std::mutex> lock(mutex_);
    if(empty_unsafe()){
        return std::nullopt;
    }
    return buffer_[tail_];
}

template <typename T>
void CircularBuffer<T>::clear(){
    std::lock_guard<std::mutex> lock(mutex_);
    head_ = 0;
    tail_ = 0;
    size_ = 0;
    is_full_ = false;
}

template <typename T>
size_t CircularBuffer<T>::size() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
}

template <typename T>
size_t CircularBuffer<T>::capacity() const{
    return capacity_;
}

template <typename T>
bool CircularBuffer<T>::is_empty() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return empty_unsafe();
}

template <typename T>
bool CircularBuffer<T>::is_full() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return is_full_;
}

template <typename T>
bool CircularBuffer<T>::empty_unsafe() const{
    return (!is_full_ && (head_ == tail_));
}
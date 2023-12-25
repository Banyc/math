pub struct VecZip<I> {
    iterators: Vec<I>,
}
impl<I> VecZip<I> {
    pub fn new(iterators: Vec<I>) -> Self {
        Self { iterators }
    }
}
impl<I: Iterator> Iterator for VecZip<I> {
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterators.iter_mut().map(Iterator::next).collect()
    }
}


// 29/09/2023
pub fn binary_search(list:&[i32], target: i32) -> usize {
    let mut bottom : usize= 0;
    let mut top = list.len() - 1;
    while top!=bottom {
        let mid = (bottom + top + 1)/2;
        println!("{top}, {bottom}, {mid}");
        if list[mid] > target {
            top = mid-1;
        }
        else {
            bottom = mid;
        }
    }
    return bottom
}

#[cfg(test)]
mod test_binary_search {
    use super::*;

    #[test]
    fn single(){
        assert_eq!(binary_search(&[1i32], 1i32), 0)
    }

    #[test]
    fn two(){
        assert_eq!(binary_search(&[1i32,2i32], 2i32), 1)
    }

    #[test]
    fn full(){
        assert_eq!(binary_search(&[1i32,2i32,3i32], 2i32), 1)
    }
}


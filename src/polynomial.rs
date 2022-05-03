use std::{fmt::Display, ops};

pub enum Error {
    Undefined,
}

pub struct Polynomial {
    coefficients: Vec<f64>, // from low-order to high-order
}

pub struct Point {
    x: f64,
    y: f64,
}

impl Polynomial {
    fn check_rep(&self) {
        if self.coefficients.len() > 0 {
            assert!(self.coefficients[self.coefficients.len() - 1] != 0.0);
        }
    }

    pub fn new(mut coefficients: Vec<f64>) -> Polynomial {
        while coefficients.len() > 0 && coefficients[coefficients.len() - 1] == 0.0 {
            coefficients.pop().unwrap();
        }
        let this = Polynomial { coefficients };
        this.check_rep();
        this
    }

    pub fn zero() -> Polynomial {
        let this = Polynomial {
            coefficients: vec![],
        };
        this.check_rep();
        this
    }

    /// find f(x) given
    ///
    /// $$ f(x) = \sum_{i=1}^n y_i \cdot (\prod_{j \ne i} \frac{x - x_j}{x_i - x_j}) $$
    pub fn interpolate(points: &Vec<Point>) -> Polynomial {
        assert!(points.len() != 0);
        // make sure $x_i$ is distinct
        for i in 0..points.len() {
            for j in 0..points.len() {
                if i == j {
                    continue;
                }
                assert!(points[i].x - points[j].x != 0.0);
            }
        }

        let mut sum = Polynomial::zero();
        for i in 0..points.len() {
            let xi = points[i].x;
            let yi = points[i].y;
            let mut prod = Polynomial::new(vec![1.0]);
            for j in 0..points.len() {
                if j == i {
                    continue;
                }
                let xj = points[j].x;
                let a0 = -xj / (xi - xj);
                let a1 = 1.0 / (xi - xj);
                let base = Polynomial::new(vec![a0, a1]);
                prod = &prod * &base;
            }
            prod = &prod * &Polynomial::new(vec![yi]);
            sum = &sum + &prod;
        }
        sum
    }

    // f: R -> R
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            result += coeff * x.powi(i as i32);
        }
        result
    }

    pub fn degree(&self) -> Result<usize, Error> {
        if self.coefficients.len() == 0 {
            Err(Error::Undefined)
        } else {
            Ok(self.coefficients.len() - 1)
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coefficients.len() == 0
    }
}

impl<'a, 'b> ops::Add<&'b Polynomial> for &'a Polynomial {
    type Output = Polynomial;

    fn add(self, rhs: &'b Polynomial) -> Self::Output {
        let len = {
            let mut max_len = self.coefficients.len();
            if rhs.coefficients.len() > max_len {
                max_len = rhs.coefficients.len();
            }
            max_len
        };
        let mut coeffs = vec![0.0; len];
        for i in 0..coeffs.len() {
            if self.coefficients.len() > i && rhs.coefficients.len() > i {
                coeffs[i] = self.coefficients[i] + rhs.coefficients[i];
            } else if self.coefficients.len() > i {
                coeffs[i] = self.coefficients[i];
            } else if rhs.coefficients.len() > i {
                coeffs[i] = rhs.coefficients[i];
            } else {
                panic!();
            }
        }
        Polynomial::new(coeffs)
    }
}

impl<'a, 'b> ops::Sub<&'b Polynomial> for &'a Polynomial {
    type Output = Polynomial;

    fn sub(self, rhs: &'b Polynomial) -> Self::Output {
        let len = {
            let mut max_len = self.coefficients.len();
            if rhs.coefficients.len() > max_len {
                max_len = rhs.coefficients.len();
            }
            max_len
        };
        let mut coeffs = vec![0.0; len];
        for i in 0..coeffs.len() {
            if self.coefficients.len() > i && rhs.coefficients.len() > i {
                coeffs[i] = self.coefficients[i] - rhs.coefficients[i];
            } else if self.coefficients.len() > i {
                coeffs[i] = self.coefficients[i];
            } else if rhs.coefficients.len() > i {
                coeffs[i] = -rhs.coefficients[i];
            } else {
                panic!();
            }
        }
        Polynomial::new(coeffs)
    }
}

impl<'a, 'b> ops::Mul<&'b Polynomial> for &'a Polynomial {
    type Output = Polynomial;

    fn mul(self, rhs: &'b Polynomial) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Polynomial::zero();
        }
        let mut coeffs = vec![0.0; self.coefficients.len() + rhs.coefficients.len() - 1];
        for (i, &i_coeff) in rhs.coefficients.iter().enumerate() {
            for (j, &j_coeff) in self.coefficients.iter().enumerate() {
                coeffs[i + j] += i_coeff * j_coeff;
            }
        }
        Polynomial::new(coeffs)
    }
}

impl Display for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        let mut first = true;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if *coeff != 0.0 {
                if !first {
                    output.push_str(" + ");
                }
                first = false;
                if i == 0 {
                    output.push_str(&format!("{}", coeff));
                } else {
                    output.push_str(&format!("{} x^{}", coeff, i));
                }
            }
        }
        write!(f, "{}", output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_evaluation() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.evaluate(0.0), 1.0);
        assert_eq!(p.evaluate(1.0), 6.0);
        assert_eq!(p.evaluate(2.0), 17.0);
    }

    #[test]
    fn test_polynomial_display() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(format!("{}", p), "1 + 2 x^1 + 3 x^2");
        let p = Polynomial::new(vec![-1.0, -2.0, 3.0]);
        assert_eq!(format!("{}", p), "-1 + -2 x^1 + 3 x^2");
    }

    #[test]
    fn test_polynomial_add() {
        let lhs = Polynomial::new(vec![1.0, 2.0, 3.0]);
        let rhs = Polynomial::new(vec![1.0, 2.0, 3.0]);
        let sum = &lhs + &rhs;
        assert_eq!(sum.coefficients, vec![2.0, 4.0, 6.0]);

        let lhs = Polynomial::new(vec![1.0, 2.0, 3.0]);
        let rhs = Polynomial::new(vec![1.0, 2.0]);
        let sum = &lhs + &rhs;
        assert_eq!(sum.coefficients, vec![2.0, 4.0, 3.0]);

        let lhs = Polynomial::new(vec![1.0]);
        let rhs = Polynomial::new(vec![1.0, 2.0]);
        let sum = &lhs + &rhs;
        assert_eq!(sum.coefficients, vec![2.0, 2.0]);

        let lhs = Polynomial::new(vec![1.0, -2.0, 3.0]);
        let rhs = Polynomial::new(vec![1.0, 1.0]);
        let sum = &lhs + &rhs;
        assert_eq!(sum.coefficients, vec![2.0, -1.0, 3.0]);
    }
    #[test]
    fn test_polynomial_mul() {
        let lhs = Polynomial::zero();
        let rhs = Polynomial::new(vec![1.0, 2.0]);
        let prod = &lhs * &rhs;
        assert_eq!(prod.coefficients, vec![]);

        let lhs = Polynomial::new(vec![1.0, 2.0]);
        let rhs = Polynomial::new(vec![1.0, 2.0]);
        let prod = &lhs * &rhs;
        assert_eq!(prod.coefficients, vec![1.0, 4.0, 4.0]);
    }

    #[test]
    fn test_polynomial_interpolate() {
        let points = vec![Point { x: 1.0, y: 1.0 }];
        let p = Polynomial::interpolate(&points);
        assert_eq!(p.coefficients, vec![1.0]);

        let points = vec![Point { x: 1.0, y: 1.0 }, Point { x: 2.0, y: 0.0 }];
        let p = Polynomial::interpolate(&points);
        assert_eq!(p.coefficients, vec![2.0, -1.0]);

        let points = vec![
            Point { x: 1.0, y: 1.0 },
            Point { x: 2.0, y: 4.0 },
            Point { x: 7.0, y: 9.0 },
        ];
        let p = Polynomial::interpolate(&points);
        assert_eq!(
            p.coefficients,
            vec![-2.666666666666666, 3.9999999999999996, -0.3333333333333334]
        );
        let ys = {
            let mut ys = Vec::new();
            for point in &points {
                ys.push(p.evaluate(point.x));
            }
            ys
        };
        assert_eq!(ys, vec![1.0, 3.999999999999999, 8.99999999999999])
    }
}

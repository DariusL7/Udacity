#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	VectorXd rmse = VectorXd(4);
	rmse << 0, 0, 0, 0;

	//Check for valid inputs.
	if(estimations.size() != ground_truth.size() || estimations.size() == 0)
	  {
	  	cout << "Invalid Inputs!" << endl;
	  	return rmse;
	  }

	//Accumulate squared residuals.
	for(unsigned int i = 0; i < estimations.size(); ++i)
	   {
	   	VectorXd sr = estimations[i] - ground_truth[i];
	   	sr = sr.array() * sr.array();
	   	rmse += sr;
	   }
	//Calculating the Mean.
	rmse = rmse/estimations.size();

	//Calculating the squared root
	rmse = rmse.array().sqrt();
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
	MatrixXd Hj = MatrixXd(3, 4);

	//State Parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float c1 = px * px + py * py;
	float c2 = sqrt(c1);
	float c3 = (c1 * c2);

	//Check division by zero
	if(fabs(c1) < 0.0001)
	  {
	  	cout << "Jacobian Division by zero!" << endl;
	  	return Hj;
	  }

	//Compute the Jacobian.
	Hj << (px / c2), (py / c2), 0, 0,
		  -(py / c1), (px / c1), 0, 0,
		  py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

    return Hj;
	
}

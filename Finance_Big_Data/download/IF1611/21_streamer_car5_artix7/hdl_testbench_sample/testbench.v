`timescale 1ns/1ns
module testbbench;
	parameter	CLK_INTERVAL = 10;

	reg			rst_n;
	reg			clk;
	reg			test_in;
	wire		test_out;

	initial begin
		forever begin
			clk	= 1'b0;
			#(CLK_INTERVAL / 2)
			clk	= 1'b1;
			#(CLK_INTERVAL / 2)
			;
		end
	end

	initial begin
		$display ("start");
		rst_n	= 1'b0;
		test_in	= 1'b0;
		#(CLK_INTERVAL)

		rst_n	= 1'b1;
		#(CLK_INTERVAL * 30)

		test_in	= 1'b1;
		#(CLK_INTERVAL * 30)

		test_in	= 1'b0;
		#(CLK_INTERVAL * 30)

		$display ("end");
		$finish;
	end

	sample_circuit u0 (
		.test_in(test_in),
		.test_out(test_out),
		.rst_n(rst_n),
		.clk(clk)
	);

endmodule

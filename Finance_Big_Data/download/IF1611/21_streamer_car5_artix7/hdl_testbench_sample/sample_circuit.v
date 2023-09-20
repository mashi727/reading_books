module sample_circuit (test_in, test_out, rst_n, clk);
	input		test_in;
	output		test_out;
	input		rst_n;
	input		clk;

	reg [1:0]	cnt_reg;

	always @(posedge clk) begin
		if (rst_n == 1'b0)
			cnt_reg	<= 2'b00;
		else if (test_in == 1'b1)
			cnt_reg	<= cnt_reg + 1'b1;
	end

	assign	test_out	= cnt_reg[1];
endmodule

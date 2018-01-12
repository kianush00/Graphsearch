package edu.umass.cs.cs646.tutorial.lucene;

import graphsearch.alpha.GTerm;

import java.util.Stack;

public class TestStack {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		Stack<GTerm> stack = new Stack<>();
		
		GTerm t1 = new GTerm("hola");
		t1.addPosition(10);
		t1.addPosition(20);
		t1.addPosition(30);
		
		System.out.println("Printing t1...");
		t1.print();
		
		GTerm t2 = new GTerm("hola");
		t2.addPosition(100);
		t2.addPosition(200);
		t2.addPosition(300);
		
		System.out.println("Printing t2...");
		t2.print();
		
		stack.push(t1);
		stack.push(t2);
		
		System.out.println("Printing stack...");
		stack.toString();
		
	}

}

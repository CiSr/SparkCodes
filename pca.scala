import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.SQLContext
import java.text.SimpleDateFormat
import java.util.Calendar
import org.apache.spark.sql.SaveMode
import org.apache.spark.rdd.RDD
import scala.annotation.tailrec
import breeze.linalg.{DenseMatrix, eigSym}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix

val df = sqlContext.sql(s""" select 
	count_ipranges_percent_suspectDevices_cut,
	count_ipranges_percent_suspectSN_cut,count_ipranges_percent_suspectproducts_cut,count_ipranges_remainingColsBucket1,
	count_ipranges_remainingColsBucket2,count_ipranges_remainingColsBucket3,count_ipranges_remainingColsBucket4,
	site_active_only_pure_iprange,percent_contracts_not_renewed,
	datediff,percent_common_devices_sitelevel from sonar_piracy_analytics_interm.site_level_score1 """) 


//Finding the mean centering


val feature_names = df.columns       

@tailrec
final def mean_centered(df: DataFrame, icount: Int ,feature_names:Array[String]):DataFrame={
if (icount == df.columns.length )df 

else	
	{
	val feature = feature_names(icount)
	val df1 = df.select(avg(feature).alias("a"))
	val mean_df = df1.select("a").collect()(0).getDouble(0)
	val temp = df.withColumn(feature+"_meaned" ,df(feature) - mean_df)
	  			 .drop(feature)
	println("The mean of " + feature + " : "+ mean_df)

	mean_centered(temp, icount+1, feature_names)   
	}

}

val final_df = mean_centered(df, 0,feature_names)

//Finding the covariance matrix 

val colNames = final_df.columns
val covariance_matrix = Array.ofDim[Double](colNames.length,colNames.length)

for(i<-0 until colNames.length){
	for(j<-0 until colNames.length)
	{
		//println("The Correlation between" +colNames(i) + colNames(j))
		val x = final_df.stat.cov(colNames(i),colNames(j))
		covariance_matrix(i)(j) = x
		covariance_matrix(j)(i) = x
			
	}
}


val vectors_rdd = sc.parallelize(covariance_matrix).map { row => val parts = row.mkString(",")
      val feature_Buffer = parts.split(",").map { ele => ele.toDouble }.toBuffer
      val feature_Vec = Vectors.dense(feature_Buffer.toArray)
      (feature_Vec)
  }


val mat: RowMatrix = new RowMatrix(vectors_rdd)


val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(5, computeU = true)
val U: RowMatrix = svd.U  
val s: Vector = svd.s     
val V: Matrix = svd.V


//To print the eigen values of all the vectors. 


s.toArray.foreach((c:Double) => println("The Eigen Values are " +c))



val dm = DenseMatrix(covariance_matrix.map(_.toArray):_*)
val eigen_values = eigSym(dm)



















/*val rdd = sc.parallelize(covariance_matrix)
	.map(Row.fromSeq(_))

val df = rdd.map({ 
  case Row(val1: Double, val2:Double, val3:Double, val4:Double, val5:Double, val6:Double, val7:Double, val8:Double, val9:Double, val10:Double, val11:Double) 
  => (val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11)
}).toDF("val1", "val2", "val3", "val4", "val5", "val6", "val7", "val8", "val9", "val10", "val11")

*/

def toRDD(m: Matrix): RDD[Vector] = {
  val columns = m.toArray.grouped(m.numRows)
  val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
  val vectors = rows.map(row => new DenseVector(row.toArray))
  sc.parallelize(vectors)
}

val y = DenseMatrix.eye(11)
val z = toRDD(y)
vectors_rdd.subtract(z).union(z.subtract(vectors_rdd))
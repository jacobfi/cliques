package com.github.jacobfi.cliques

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.annotation.tailrec

object CliqueFinder {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    val input = spark.read.option("header", "false").csv("src/main/resources/example.csv").toDF("from", "to")
    val edges = input.unionByName(input.select($"to" as "from", $"from" as "to")).cache()

    val vertices = edges.select($"from" as "id").distinct().cache()

    val root = vertices.select(collect_list($"id") as "cand", array() as "compsub", array() as "not")

    cliqueEnumerate(root).show()

    @tailrec
    def cliqueEnumerate(tree: DataFrame): DataFrame = {
      val filtered = tree.filter(size($"cand") > 0)
      if (filtered.isEmpty) tree
      else {
        val base = filtered.withColumn("node", monotonically_increasing_id()).cache()

        // TODO: More clean/efficient method for finding fixpoint.
        val fixpoint = base.withColumn("v", explode($"cand"))
            .join(edges, $"v" === $"from")
            .unionByName(base.withColumn("v", explode($"cand")).withColumn("from", $"v").withColumn("to", $"v"))
            .filter(array_contains($"cand", $"to"))
            .groupBy("node", "v").agg(count("to") as "count")
            .orderBy($"count".desc)
            .take(1).head.getAs[String]("v")

        // TODO: Replace operations on arrays with operations directly on the DataFrame.
        val nodeWindow = Window.partitionBy("node").orderBy("v")
        val expanded = base
            .select($"node", explode($"cand") as "v", lit(fixpoint) as "fixp")
            .join(edges, $"v" === $"from")
            .groupBy("node", "v", "fixp").agg(collect_list("to") as "adj")
            .filter(!array_contains($"adj", $"fixp"))
            .join(base, "node")
            .select(
              $"v", $"adj", $"compsub",
              array_union($"not", collect_list("v").over(nodeWindow)) as "not",
              array_except($"cand", collect_list("v").over(nodeWindow)) as "cand"
            )
            .select(
              array_union($"compsub", array($"v")) as "compsub", // new_cs <- compsub + cur_v
              array_intersect($"cand", $"adj") as "cand", // new_cand <- cand intersect adj
              array_intersect($"not", $"adj") as "not" // new_not <- not intersect adj
            )
            .cache()

        expanded.show()

        expanded.rdd.localCheckpoint()
        expanded.foreachPartition(_ => Unit)

        tree.unpersist()
        base.unpersist()

        cliqueEnumerate(expanded)
      }
    }

  }

}
